import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics, ensemble
import lightgbm as lgb
import itertools
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import gc
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn import metrics

#Loading the data
train_df = pd.read_csv("train_cm.csv")
test_df = pd.read_csv("test_cm.csv")
train_df.dtypes
test_df.dtypes

#Removing outliers/data clipping
train_df[train_df['roomnights']<0]=0
test_df[test_df['roomnights']<0]=0

#Date formatting
train_df["booking_date"]= pd.to_datetime(train_df["booking_date"],dayfirst=True)
train_df["checkin_date"]= pd.to_datetime(train_df["checkin_date"],dayfirst=True)
train_df["checkout_date"]= pd.to_datetime(train_df["checkout_date"],dayfirst=True)
train_df.dtypes
test_df.dtypes
gc.collect()
gc.collect()

#Listing out the available features
ID_y_col = ["reservation_id","amount_spent_per_room_night_scaled"]
raw_col = [col for col in train_df.columns if col not in ID_y_col]
train_df["train_set"] = 1
test_df["train_set"] = 0
test_df["amount_spent_per_room_night_scaled"] = -9

#Concat Train and Test data for creating different features as both Train and Test have similar distribution
all_df = pd.concat([train_df, test_df])
gc.collect()
gc.collect()

#Dates based features
all_df['b_yr'] = all_df['booking_date'].dt.year
all_df['b_mon'] = all_df['booking_date'].dt.month
all_df['b_week'] = all_df['booking_date'].dt.week
all_df['b_day'] = all_df['booking_date'].dt.dayofweek

all_df['cin_yr'] = all_df['checkin_date'].dt.year
all_df['cin_mon'] = all_df['checkin_date'].dt.month
all_df['cin_week'] = all_df['checkin_date'].dt.week
all_df['cin_day'] = all_df['checkin_date'].dt.dayofweek

all_df['cout_yr'] = all_df['checkout_date'].dt.year
all_df['cout_mon'] = all_df['checkout_date'].dt.month
all_df['cout_week'] = all_df['checkout_date'].dt.week
all_df['cout_day'] = all_df['checkout_date'].dt.dayofweek

#Data Imputation
all_df[['season_holidayed_code']] = all_df[['season_holidayed_code']].fillna(value=5)
all_df[['state_code_residence']] = all_df[['state_code_residence']].fillna(value=39)
all_df['season_holidayed_code'] = all_df['season_holidayed_code'].round(0)
all_df['state_code_residence'] = all_df['state_code_residence'].round(0)

#Days taken between Booking and Checkin, Checkout and Checkin
all_df['diff_cin_b']= (all_df['checkin_date'] - all_df['booking_date']).dt.days
all_df['diff_cout_cin']= (all_df['checkout_date'] - all_df['checkin_date']).dt.days

#Label Encoding of categorical variables
train_df = all_df[all_df["train_set"]==1].reset_index(drop=True)
test_df = all_df[all_df["train_set"]==0].reset_index(drop=True)
le_col = ["member_age_buckets","memberid", "cluster_code","reservationstatusid_code","resort_id"]
indexer = {}
for col in tqdm(le_col):
    if col == 'reservation_id': continue
    _, indexer[col] = pd.factorize(all_df[col])

for col in tqdm(le_col):
    if col == 'reservation_id': continue
    train_df[col] = indexer[col].get_indexer(train_df[col])
    test_df[col] = indexer[col].get_indexer(test_df[col])
gc.collect()
gc.collect()
train_df.dtypes

all_df = pd.concat([train_df, test_df])

# Number of reservations made by customers w.r.t various user/resort/booking based categories/characteristics
gdf = all_df.groupby(["channel_code"])["reservation_id"].count().reset_index()
gdf.columns = ["channel_code", "chan_count"]
all_df = all_df.merge(gdf, on=["channel_code"], how="left")

gdf = all_df.groupby(["main_product_code"])["reservation_id"].count().reset_index()
gdf.columns = ["main_product_code", "prd_count"]
all_df = all_df.merge(gdf, on=["main_product_code"], how="left")

gdf = all_df.groupby(["persontravellingid"])["reservation_id"].count().reset_index()
gdf.columns = ["persontravellingid", "pers_trav_count"]
all_df = all_df.merge(gdf, on=["persontravellingid"], how="left")

gdf = all_df.groupby(["resort_region_code"])["reservation_id"].count().reset_index()
gdf.columns = ["resort_region_code", "resort_reg_count"]
all_df = all_df.merge(gdf, on=["resort_region_code"], how="left")

gdf = all_df.groupby(["resort_type_code"])["reservation_id"].count().reset_index()
gdf.columns = ["resort_type_code", "resort_typ_count"]
all_df = all_df.merge(gdf, on=["resort_type_code"], how="left")

gdf = all_df.groupby(["room_type_booked_code"])["reservation_id"].count().reset_index()
gdf.columns = ["room_type_booked_code", "room_typ_count"]
all_df = all_df.merge(gdf, on=["room_type_booked_code"], how="left")

gdf = all_df.groupby(["season_holidayed_code"])["reservation_id"].count().reset_index()
gdf.columns = ["season_holidayed_code", "seas_count"]
all_df = all_df.merge(gdf, on=["season_holidayed_code"], how="left")

gdf = all_df.groupby(["state_code_residence"])["reservation_id"].count().reset_index()
gdf.columns = ["state_code_residence", "state_res_count"]
all_df = all_df.merge(gdf, on=["state_code_residence"], how="left")

gdf = all_df.groupby(["state_code_resort"])["reservation_id"].count().reset_index()
gdf.columns = ["state_code_resort", "state_resort_count"]
all_df = all_df.merge(gdf, on=["state_code_resort"], how="left")

gdf = all_df.groupby(["member_age_buckets"])["reservation_id"].count().reset_index()
gdf.columns = ["member_age_buckets", "age_count"]
all_df = all_df.merge(gdf, on=["member_age_buckets"], how="left")

gdf = all_df.groupby(["booking_type_code"])["reservation_id"].count().reset_index()
gdf.columns = ["booking_type_code", "book_type_count"]
all_df = all_df.merge(gdf, on=["booking_type_code"], how="left")

gdf = all_df.groupby(["memberid"])["reservation_id"].count().reset_index()
gdf.columns = ["memberid", "mem_id_count"]
all_df = all_df.merge(gdf, on=["memberid"], how="left")

gdf = all_df.groupby(["cluster_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cluster_code", "cluster_count"]
all_df = all_df.merge(gdf, on=["cluster_code"], how="left")

gdf = all_df.groupby(["reservationstatusid_code"])["reservation_id"].count().reset_index()
gdf.columns = ["reservationstatusid_code", "reserv_status_count"]
all_df = all_df.merge(gdf, on=["reservationstatusid_code"], how="left")

gdf = all_df.groupby(["resort_id"])["reservation_id"].count().reset_index()
gdf.columns = ["resort_id", "resort_id_count"]
all_df = all_df.merge(gdf, on=["resort_id"], how="left")

gdf = all_df.groupby(["numberofadults"])["reservation_id"].count().reset_index()
gdf.columns = ["numberofadults", "adult_count"]
all_df = all_df.merge(gdf, on=["numberofadults"], how="left")

gdf = all_df.groupby(["numberofchildren"])["reservation_id"].count().reset_index()
gdf.columns = ["numberofchildren", "child_count"]
all_df = all_df.merge(gdf, on=["numberofchildren"], how="left")

gdf = all_df.groupby(["roomnights"])["reservation_id"].count().reset_index()
gdf.columns = ["roomnights", "nights_count"]
all_df = all_df.merge(gdf, on=["roomnights"], how="left")

gdf = all_df.groupby(["total_pax"])["reservation_id"].count().reset_index()
gdf.columns = ["total_pax", "pax_count"]
all_df = all_df.merge(gdf, on=["total_pax"], how="left")

#Intutive Features based on numeric data
all_df['t_mem']= all_df['numberofadults'] + all_df['numberofchildren']

all_df['r_child_adult']= all_df['numberofchildren']/all_df['numberofadults']
all_df['r_child_adult'] = all_df['r_child_adult'].round(2)

all_df['r_room_night']= all_df['roomnights']/all_df['diff_cout_cin']
all_df['r_room_night'] = all_df['r_room_night'].round(2)

all_df['r_pax_room_night']= all_df['total_pax']/all_df['roomnights']
all_df['r_pax_room_night'] = all_df['r_pax_room_night'].round(2)

all_df['r_t_mem_room_night']= all_df['t_mem']/all_df['roomnights']
all_df['r_t_mem_room_night'] = all_df['r_pax_room_night'].round(2)

all_df['r_child_room_night']= all_df['numberofchildren']/all_df['roomnights']
all_df['r_child_room_night'] = all_df['r_child_room_night'].round(2)

all_df['r_adult_room_night']= all_df['numberofadults']/all_df['roomnights']
all_df['r_adult_room_night'] = all_df['r_adult_room_night'].round(2)

#Number of reservations made by customers w.r.t their checkin dates
gdf = all_df.groupby(["cin_yr"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_yr", "cin_yr_count"]
all_df = all_df.merge(gdf, on=["cin_yr"], how="left")

gdf = all_df.groupby(["cin_mon"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_mon", "cin_mon_count"]
all_df = all_df.merge(gdf, on=["cin_mon"], how="left")

gdf = all_df.groupby(["cin_week"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_week", "cin_week_count"]
all_df = all_df.merge(gdf, on=["cin_week"], how="left")

gdf = all_df.groupby(["cin_day"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_day", "cin_day_count"]
all_df = all_df.merge(gdf, on=["cin_day"], how="left")

#Number of reservations made by customers w.r.t their checkout dates
gdf = all_df.groupby(["cout_yr"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_yr", "cout_yr_count"]
all_df = all_df.merge(gdf, on=["cout_yr"], how="left")

gdf = all_df.groupby(["cout_mon"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_mon", "cout_mon_count"]
all_df = all_df.merge(gdf, on=["cout_mon"], how="left")

gdf = all_df.groupby(["cout_week"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_week", "cout_week_count"]
all_df = all_df.merge(gdf, on=["cout_week"], how="left")

gdf = all_df.groupby(["cout_day"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_day", "cout_day_count"]
all_df = all_df.merge(gdf, on=["cout_day"], how="left")

#Number of reservations made by customers w.r.t their booking dates
gdf = all_df.groupby(["b_yr"])["reservation_id"].count().reset_index()
gdf.columns = ["b_yr", "b_yr_count"]
all_df = all_df.merge(gdf, on=["b_yr"], how="left")

gdf = all_df.groupby(["b_mon"])["reservation_id"].count().reset_index()
gdf.columns = ["b_mon", "b_mon_count"]
all_df = all_df.merge(gdf, on=["b_mon"], how="left")

gdf = all_df.groupby(["b_week"])["reservation_id"].count().reset_index()
gdf.columns = ["b_week", "b_week_count"]
all_df = all_df.merge(gdf, on=["b_week"], how="left")

gdf = all_df.groupby(["b_day"])["reservation_id"].count().reset_index()
gdf.columns = ["b_day", "b_day_count"]
all_df = all_df.merge(gdf, on=["b_day"], how="left")

#Descriptive stats of adults/children/total_pax who have stayed w.r.t Number of roomnights booked, persontravellingid, resort_type_code, state_code_resort, booking_type_code, cluster_code, season_holidayed_code with/without checkin dates
##Descriptive stats of adults who have stayed w.r.t Number of roomnights booked
gdf = all_df.groupby(["roomnights"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["roomnights", "nights_adults_min", "nights_adults_max", "nights_adults_mean", "nights_adults_std"]
all_df = all_df.merge(gdf, on=["roomnights"], how="left")

##Descriptive stats of adults who have stayed w.r.t Number of roomnights booked, checkin dates
gdf = all_df.groupby(["cin_mon", "cin_day", "roomnights"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "roomnights", "cin_m_d_nights_adults_min", "cin_m_d_nights_adults_max", "cin_m_d_nights_adults_mean", "cin_m_d_nights_adults_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "roomnights"], how="left")

#Descriptive stats of children who have stayed w.r.t Number of roomnights booked (as it may have direct impact on expenses)
gdf = all_df.groupby(["roomnights"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["roomnights", "nights_child_min", "nights_child_max", "nights_child_mean", "nights_child_std"]
all_df = all_df.merge(gdf, on=["roomnights"], how="left")

##Descriptive stats of children who have stayed w.r.t Number of roomnights booked, checkin dates
gdf = all_df.groupby(["cin_mon", "cin_day", "roomnights"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "roomnights", "cin_m_d_nights_child_min", "cin_m_d_nights_child_max", "cin_m_d_nights_child_mean", "cin_m_d_nights_child_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "roomnights"], how="left")

#Descriptive stats of total_pax who have stayed w.r.t Number of roomnights booked
gdf = all_df.groupby(["roomnights"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["roomnights", "nights_pax_min", "nights_pax_max", "nights_pax_mean", "nights_pax_std"]
all_df = all_df.merge(gdf, on=["roomnights"], how="left")

##Descriptive stats of total_pax who have stayed w.r.t Number of roomnights booked, checkin dates
gdf = all_df.groupby(["cin_mon", "cin_day", "roomnights"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "roomnights", "cin_m_d_nights_pax_min", "cin_m_d_nights_pax_max", "cin_m_d_nights_pax_mean", "cin_m_d_nights_pax_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "roomnights"], how="left")

#Descriptive stats of adults who have stayed w.r.t persontravellingid
gdf = all_df.groupby(["persontravellingid"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["persontravellingid", "personid_adults_min", "personid_adults_max", "personid_adults_mean", "personid_adults_std"]
all_df = all_df.merge(gdf, on=["persontravellingid"], how="left")

##Descriptive stats of adults who have stayed w.r.t persontravellingid, checkin dates
gdf = all_df.groupby(["cin_mon", "cin_day", "persontravellingid"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "persontravellingid", "cin_m_d_person_adults_min", "cin_m_d_person_adults_max", "cin_m_d_person_adults_mean", "cin_m_d_person_adults_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "persontravellingid"], how="left")

#Descriptive stats of children who have stayed w.r.t persontravellingid
gdf = all_df.groupby(["persontravellingid"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["persontravellingid", "personid_child_min", "personid_child_max", "personid_child_mean", "personid_child_std"]
all_df = all_df.merge(gdf, on=["persontravellingid"], how="left")

##Descriptive stats of children who have stayed w.r.t persontravellingid, checkin dates
gdf = all_df.groupby(["cin_mon", "cin_day", "persontravellingid"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "persontravellingid", "cin_m_d_person_child_min", "cin_m_d_person_child_max", "cin_m_d_person_child_mean", "cin_m_d_person_child_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "persontravellingid"], how="left")

###
gdf = all_df.groupby(["persontravellingid"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["persontravellingid", "personid_pax_min", "personid_pax_max", "personid_pax_mean", "personid_pax_std"]
all_df = all_df.merge(gdf, on=["persontravellingid"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "persontravellingid"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "persontravellingid", "cin_m_d_person_pax_min", "cin_m_d_person_pax_max", "cin_m_d_person_pax_mean", "cin_m_d_person_pax_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "persontravellingid"], how="left")

##
gdf = all_df.groupby(["resort_type_code"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["resort_type_code", "resort_typ_adults_min", "resort_typ_adults_max", "resort_typ_adults_mean", "resort_typ_adults_std"]
all_df = all_df.merge(gdf, on=["resort_type_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "resort_type_code"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "resort_type_code", "cin_m_d_resort_typ_adults_min", "cin_m_d_resort_typ_adults_max", "cin_m_d_resort_typ_adults_mean", "cin_m_d_resort_typ_adults_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "resort_type_code"], how="left")

###
gdf = all_df.groupby(["resort_type_code"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["resort_type_code", "resort_typ_child_min", "resort_typ_child_max", "resort_typ_child_mean", "resort_typ_child_std"]
all_df = all_df.merge(gdf, on=["resort_type_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "resort_type_code"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "resort_type_code", "cin_m_d_resort_typ_child_min", "cin_m_d_resort_typ_child_max", "cin_m_d_resort_typ_child_mean", "cin_m_d_resort_typ_child_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "resort_type_code"], how="left")

###
gdf = all_df.groupby(["resort_type_code"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["resort_type_code", "resort_typ_pax_min", "resort_typ_pax_max", "resort_typ_pax_mean", "resort_typ_pax_std"]
all_df = all_df.merge(gdf, on=["resort_type_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "resort_type_code"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "resort_type_code", "cin_m_d_resort_typ_pax_min", "cin_m_d_resort_typ_pax_max", "cin_m_d_resort_typ_pax_mean", "cin_m_d_resort_typ_pax_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "resort_type_code"], how="left")

###
gdf = all_df.groupby(["state_code_resort"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["state_code_resort", "state_resort_adults_min", "state_resort_adults_max", "state_resort_adults_mean", "state_resort_adults_std"]
all_df = all_df.merge(gdf, on=["state_code_resort"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "state_code_resort"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "state_code_resort", "cin_m_d_state_resort_adults_min", "cin_m_d_state_resort_adults_max", "cin_m_d_state_resort_adults_mean", "cin_m_d_state_resort_adults_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "state_code_resort"], how="left")

###
gdf = all_df.groupby(["state_code_resort"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["state_code_resort", "state_resort_child_min", "state_resort_child_max", "state_resort_child_mean", "state_resort_child_std"]
all_df = all_df.merge(gdf, on=["state_code_resort"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "state_code_resort"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "state_code_resort", "cin_m_d_state_resort_child_min", "cin_m_d_state_resort_child_max", "cin_m_d_state_resort_child_mean", "cin_m_d_state_resort_child_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "state_code_resort"], how="left")

###
gdf = all_df.groupby(["state_code_resort"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["state_code_resort", "state_resort_pax_min", "state_resort_pax_max", "state_resort_pax_mean", "state_resort_pax_std"]
all_df = all_df.merge(gdf, on=["state_code_resort"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "state_code_resort"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "state_code_resort", "cin_m_d_state_resort_pax_min", "cin_m_d_state_resort_pax_max", "cin_m_d_state_resort_pax_mean", "cin_m_d_state_resort_pax_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "state_code_resort"], how="left")
###
gdf = all_df.groupby(["booking_type_code"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["booking_type_code", "book_typ_adults_min", "book_typ_adults_max", "book_typ_adults_mean", "book_typ_adults_std"]
all_df = all_df.merge(gdf, on=["booking_type_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "booking_type_code"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "booking_type_code", "cin_m_d_book_typ_adults_min", "cin_m_d_book_typ_adults_max", "cin_m_d_book_typ_adults_mean", "cin_m_d_book_typ_adults_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "booking_type_code"], how="left")

###
gdf = all_df.groupby(["booking_type_code"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["booking_type_code", "book_typ_child_min", "book_typ_child_max", "book_typ_child_mean", "book_typ_child_std"]
all_df = all_df.merge(gdf, on=["booking_type_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "booking_type_code"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "booking_type_code", "cin_m_d_book_typ_child_min", "cin_m_d_book_typ_child_max", "cin_m_d_book_typ_child_mean", "cin_m_d_book_typ_child_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "booking_type_code"], how="left")

###
gdf = all_df.groupby(["booking_type_code"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["booking_type_code", "book_typ_pax_min", "book_typ_pax_max", "book_typ_pax_mean", "book_typ_pax_std"]
all_df = all_df.merge(gdf, on=["booking_type_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "booking_type_code"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "booking_type_code", "cin_m_d_book_typ_pax_min", "cin_m_d_book_typ_pax_max", "cin_m_d_book_typ_pax_mean", "cin_m_d_book_typ_pax_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "booking_type_code"], how="left")

###
gdf = all_df.groupby(["cluster_code"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cluster_code", "cluster_adults_min", "cluster_adults_max", "cluster_adults_mean", "cluster_adults_std"]
all_df = all_df.merge(gdf, on=["cluster_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "cluster_code"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "cluster_code", "cin_m_d_cluster_adults_min", "cin_m_d_cluster_adults_max", "cin_m_d_cluster_adults_mean", "cin_m_d_cluster_adults_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "cluster_code"], how="left")

###
gdf = all_df.groupby(["cluster_code"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cluster_code", "cluster_child_min", "cluster_child_max", "cluster_child_mean", "cluster_child_std"]
all_df = all_df.merge(gdf, on=["cluster_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "cluster_code"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "cluster_code", "cin_m_d_cluster_child_min", "cin_m_d_cluster_child_max", "cin_m_d_cluster_child_mean", "cin_m_d_cluster_child_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "cluster_code"], how="left")

###
gdf = all_df.groupby(["cluster_code"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cluster_code", "cluster_pax_min", "cluster_pax_max", "cluster_pax_mean", "cluster_pax_std"]
all_df = all_df.merge(gdf, on=["cluster_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_day", "cluster_code"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_mon", "cin_day", "cluster_code", "cin_m_d_cluster_pax_min", "cin_m_d_cluster_pax_max", "cin_m_d_cluster_pax_mean", "cin_m_d_cluster_pax_std"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "cluster_code"], how="left")

###
gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_adults_min", "cin_day_mon_nights_adults_max", "cin_day_mon_nights_adults_mean", "cin_day_mon_nights_adults_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_child_min", "cin_day_mon_nights_child_max", "cin_day_mon_nights_child_mean", "cin_day_mon_nights_child_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_pax_min", "cin_day_mon_nights_pax_max", "cin_day_mon_nights_pax_mean", "cin_day_mon_nights_pax_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

####
gdf = all_df.groupby(["season_holidayed_code", "cin_day", "cin_mon", "roomnights"])["numberofadults"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["season_holidayed_code", "cin_day", "cin_mon","roomnights", "seas_cin_day_mon_nights_adults_min", "seas_cin_day_mon_nights_adults_max", "seas_cin_day_mon_nights_adults_mean", "seas_cin_day_mon_nights_adults_std"]
all_df = all_df.merge(gdf, on=["season_holidayed_code", "cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["season_holidayed_code", "cin_day", "cin_mon", "roomnights"])["numberofchildren"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["season_holidayed_code", "cin_day", "cin_mon","roomnights", "seas_cin_day_mon_nights_child_min", "seas_cin_day_mon_nights_child_max", "seas_cin_day_mon_nights_child_mean", "seas_cin_day_mon_nights_child_std"]
all_df = all_df.merge(gdf, on=["season_holidayed_code", "cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["season_holidayed_code", "cin_day", "cin_mon", "roomnights"])["total_pax"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["season_holidayed_code", "cin_day", "cin_mon","roomnights", "seas_cin_day_mon_nights_pax_min", "seas_cin_day_mon_nights_pax_max", "seas_cin_day_mon_nights_pax_mean", "seas_cin_day_mon_nights_pax_std"]
all_df = all_df.merge(gdf, on=["season_holidayed_code", "cin_day", "cin_mon", "roomnights"], how="left")
####

#ordinal value of dates inorder to calculate shifts
all_df['checkin_date_ord'] = all_df['checkin_date'].apply(lambda x: x.toordinal())
all_df['booking_date_ord'] = all_df['booking_date'].apply(lambda x: x.toordinal())

#Sorting of data
all_df = all_df.sort_values(by=["memberid", "checkin_date_ord"]).reset_index(drop=True)

#Selected 4 shifts as 75% of members have max of 4 visits
#Memberid wise what were the Previous 4 checkins and Next 4 checkins
all_df["cin_shift1"] = all_df.groupby("memberid")["checkin_date_ord"].shift(1)
all_df["cin_shift2"] = all_df.groupby("memberid")["checkin_date_ord"].shift(2)
all_df["cin_shift3"] = all_df.groupby("memberid")["checkin_date_ord"].shift(3)
all_df["cin_shift4"] = all_df.groupby("memberid")["checkin_date_ord"].shift(4)
all_df["cin_shiftn1"] = all_df.groupby("memberid")["checkin_date_ord"].shift(-1)
all_df["cin_shiftn2"] = all_df.groupby("memberid")["checkin_date_ord"].shift(-2)
all_df["cin_shiftn3"] = all_df.groupby("memberid")["checkin_date_ord"].shift(-3)
all_df["cin_shiftn4"] = all_df.groupby("memberid")["checkin_date_ord"].shift(-4)

#Memberid wise days between current checkin and previous 4 checkins
all_df["diff1_cin"] = all_df['checkin_date_ord'] - all_df['cin_shift1']
all_df["diff2_cin"] = all_df['checkin_date_ord'] - all_df['cin_shift2']
all_df["diff3_cin"] = all_df['checkin_date_ord'] - all_df['cin_shift3']
all_df["diff4_cin"] = all_df['checkin_date_ord'] - all_df['cin_shift4']

#Memberid wise days between current checkin and next 4 checkins
all_df["diffn1_cin"] = all_df['checkin_date_ord'] - all_df['cin_shiftn1']
all_df["diffn2_cin"] = all_df['checkin_date_ord'] - all_df['cin_shiftn2']
all_df["diffn3_cin"] = all_df['checkin_date_ord'] - all_df['cin_shiftn3']
all_df["diffn4_cin"] = all_df['checkin_date_ord'] - all_df['cin_shiftn4']

#Memberid wise what were the Previous 4 bookings and Next 4 bookings
all_df = all_df.sort_values(by=["memberid", "booking_date_ord"]).reset_index(drop=True)
all_df["b_shift1"] = all_df.groupby("memberid")["booking_date_ord"].shift(1)
all_df["b_shift2"] = all_df.groupby("memberid")["booking_date_ord"].shift(2)
all_df["b_shift3"] = all_df.groupby("memberid")["booking_date_ord"].shift(3)
all_df["b_shift4"] = all_df.groupby("memberid")["booking_date_ord"].shift(4)
all_df["b_shiftn1"] = all_df.groupby("memberid")["booking_date_ord"].shift(-1)
all_df["b_shiftn2"] = all_df.groupby("memberid")["booking_date_ord"].shift(-2)
all_df["b_shiftn3"] = all_df.groupby("memberid")["booking_date_ord"].shift(-3)
all_df["b_shiftn4"] = all_df.groupby("memberid")["booking_date_ord"].shift(-4)

#Memberid wise days between current bookings and previous/next 4 bookings
all_df["diff1_b"] = all_df['booking_date_ord'] - all_df['b_shift1']
all_df["diff2_b"] = all_df['booking_date_ord'] - all_df['b_shift2']
all_df["diff3_b"] = all_df['booking_date_ord'] - all_df['b_shift3']
all_df["diff4_b"] = all_df['booking_date_ord'] - all_df['b_shift4']
all_df["diffn1_b"] = all_df['booking_date_ord'] - all_df['b_shiftn1']
all_df["diffn2_b"] = all_df['booking_date_ord'] - all_df['b_shiftn2']
all_df["diffn3_b"] = all_df['booking_date_ord'] - all_df['b_shiftn3']
all_df["diffn4_b"] = all_df['booking_date_ord'] - all_df['b_shiftn4']

#Memberid wise what were the Previous 3 resort_types/resorts_region visited, roomnights booked, seasons visited, total_pax, checkin_months, booking_months, resort_ids
all_df = all_df.sort_values(by=["memberid", "checkin_date_ord"]).reset_index(drop=True)
all_df["resort_typ_shift1"] = all_df.groupby("memberid")["resort_type_code"].shift(1)
all_df["resort_typ_shift2"] = all_df.groupby("memberid")["resort_type_code"].shift(2)
all_df["resort_typ_shift3"] = all_df.groupby("memberid")["resort_type_code"].shift(3)

all_df["resort_reg_shift1"] = all_df.groupby("memberid")["resort_region_code"].shift(1)
all_df["resort_reg_shift2"] = all_df.groupby("memberid")["resort_region_code"].shift(2)
all_df["resort_reg_shift3"] = all_df.groupby("memberid")["resort_region_code"].shift(3)

all_df["roomnights_shift1"] = all_df.groupby("memberid")["roomnights"].shift(1)
all_df["roomnights_shift2"] = all_df.groupby("memberid")["roomnights"].shift(2)
all_df["roomnights_shift3"] = all_df.groupby("memberid")["roomnights"].shift(3)

all_df["seas_shift1"] = all_df.groupby("memberid")["season_holidayed_code"].shift(1)
all_df["seas_shift2"] = all_df.groupby("memberid")["season_holidayed_code"].shift(2)
all_df["seas_shift3"] = all_df.groupby("memberid")["season_holidayed_code"].shift(3)

all_df["pax_shift1"] = all_df.groupby("memberid")["total_pax"].shift(1)
all_df["pax_shift2"] = all_df.groupby("memberid")["total_pax"].shift(2)
all_df["pax_shift3"] = all_df.groupby("memberid")["total_pax"].shift(3)

all_df["cin_mon_shift1"] = all_df.groupby("memberid")["cin_mon"].shift(1)
all_df["cin_mon_shift2"] = all_df.groupby("memberid")["cin_mon"].shift(2)
all_df["cin_mon_shift3"] = all_df.groupby("memberid")["cin_mon"].shift(3)

all_df["b_mon_shift1"] = all_df.groupby("memberid")["b_mon"].shift(1)
all_df["b_mon_shift2"] = all_df.groupby("memberid")["b_mon"].shift(2)
all_df["b_mon_shift3"] = all_df.groupby("memberid")["b_mon"].shift(3)

all_df["resort_id_shift1"] = all_df.groupby("memberid")["resort_id"].shift(1)
all_df["resort_id_shift2"] = all_df.groupby("memberid")["resort_id"].shift(2)
all_df["resort_id_shift3"] = all_df.groupby("memberid")["resort_id"].shift(3)

#Sorting of data
all_df = all_df.sort_values(by=["memberid", "checkin_date_ord"]).reset_index(drop=True)

#Memberid wise what were the Next 3 resort_types/resorts_region visited, roomnights booked, seasons visited, total_pax, checkin_months, booking_months, resort_ids
all_df["resort_typ_shiftn1"] = all_df.groupby("memberid")["resort_type_code"].shift(-1)
all_df["resort_typ_shiftn2"] = all_df.groupby("memberid")["resort_type_code"].shift(-2)
all_df["resort_typ_shiftn3"] = all_df.groupby("memberid")["resort_type_code"].shift(-3)

all_df["resort_reg_shiftn1"] = all_df.groupby("memberid")["resort_region_code"].shift(-1)
all_df["resort_reg_shiftn2"] = all_df.groupby("memberid")["resort_region_code"].shift(-2)
all_df["resort_reg_shiftn3"] = all_df.groupby("memberid")["resort_region_code"].shift(-3)

all_df["roomnights_shiftn1"] = all_df.groupby("memberid")["roomnights"].shift(-1)
all_df["roomnights_shiftn2"] = all_df.groupby("memberid")["roomnights"].shift(-2)
all_df["roomnights_shiftn3"] = all_df.groupby("memberid")["roomnights"].shift(-3)

all_df["seas_shiftn1"] = all_df.groupby("memberid")["season_holidayed_code"].shift(-1)
all_df["seas_shiftn2"] = all_df.groupby("memberid")["season_holidayed_code"].shift(-2)
all_df["seas_shiftn3"] = all_df.groupby("memberid")["season_holidayed_code"].shift(-3)

all_df["pax_shiftn1"] = all_df.groupby("memberid")["total_pax"].shift(-1)
all_df["pax_shiftn2"] = all_df.groupby("memberid")["total_pax"].shift(-2)
all_df["pax_shiftn3"] = all_df.groupby("memberid")["total_pax"].shift(-3)

all_df["cin_mon_shiftn1"] = all_df.groupby("memberid")["cin_mon"].shift(-1)
all_df["cin_mon_shiftn2"] = all_df.groupby("memberid")["cin_mon"].shift(-2)
all_df["cin_mon_shiftn3"] = all_df.groupby("memberid")["cin_mon"].shift(-3)

all_df["b_mon_shiftn1"] = all_df.groupby("memberid")["b_mon"].shift(-1)
all_df["b_mon_shiftn2"] = all_df.groupby("memberid")["b_mon"].shift(-2)
all_df["b_mon_shiftn3"] = all_df.groupby("memberid")["b_mon"].shift(-3)

all_df["resort_id_shiftn1"] = all_df.groupby("memberid")["resort_id"].shift(-1)
all_df["resort_id_shiftn2"] = all_df.groupby("memberid")["resort_id"].shift(-2)
all_df["resort_id_shiftn3"] = all_df.groupby("memberid")["resort_id"].shift(-3)

#Number of reservations made w.r.t checkin_out_month_week_day and various other combinations of categories
gdf = all_df.groupby(["cout_mon", "resort_region_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_mon","resort_region_code", "cout_mon_resort_reg_count"]
all_df = all_df.merge(gdf, on=["cout_mon","resort_region_code"], how="left")

gdf = all_df.groupby(["cout_mon", "resort_type_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_mon","resort_type_code", "cout_mon_resort_typ_count"]
all_df = all_df.merge(gdf, on=["cout_mon","resort_type_code"], how="left")

gdf = all_df.groupby(["cout_mon", "season_holidayed_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_mon","season_holidayed_code", "cout_mon_seas_count"]
all_df = all_df.merge(gdf, on=["cout_mon","season_holidayed_code"], how="left")

gdf = all_df.groupby(["cout_mon", "cluster_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_mon","cluster_code", "cout_mon_clust_count"]
all_df = all_df.merge(gdf, on=["cout_mon","cluster_code"], how="left")

gdf = all_df.groupby(["cin_mon", "resort_region_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_mon","resort_region_code", "cin_mon_resort_reg_count"]
all_df = all_df.merge(gdf, on=["cin_mon","resort_region_code"], how="left")

gdf = all_df.groupby(["cin_mon", "resort_type_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_mon","resort_type_code", "cin_mon_resort_typ_count"]
all_df = all_df.merge(gdf, on=["cin_mon","resort_type_code"], how="left")

gdf = all_df.groupby(["cin_mon", "season_holidayed_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_mon","season_holidayed_code", "cin_mon_seas_count"]
all_df = all_df.merge(gdf, on=["cin_mon","season_holidayed_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cluster_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_mon","cluster_code", "cin_mon_clust_count"]
all_df = all_df.merge(gdf, on=["cin_mon","cluster_code"], how="left")
###
gdf = all_df.groupby(["cout_week", "resort_region_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_week","resort_region_code", "cout_week_resort_reg_count"]
all_df = all_df.merge(gdf, on=["cout_week","resort_region_code"], how="left")

gdf = all_df.groupby(["cout_week", "resort_type_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_week","resort_type_code", "cout_week_resort_typ_count"]
all_df = all_df.merge(gdf, on=["cout_week","resort_type_code"], how="left")

gdf = all_df.groupby(["cout_week", "season_holidayed_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_week","season_holidayed_code", "cout_week_seas_count"]
all_df = all_df.merge(gdf, on=["cout_week","season_holidayed_code"], how="left")

gdf = all_df.groupby(["cout_week", "cluster_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_week","cluster_code", "cout_week_clust_count"]
all_df = all_df.merge(gdf, on=["cout_week","cluster_code"], how="left")

gdf = all_df.groupby(["cin_week", "resort_region_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_week","resort_region_code", "cin_week_resort_reg_count"]
all_df = all_df.merge(gdf, on=["cin_week","resort_region_code"], how="left")

gdf = all_df.groupby(["cin_week", "resort_type_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_week","resort_type_code", "cin_week_resort_typ_count"]
all_df = all_df.merge(gdf, on=["cin_week","resort_type_code"], how="left")

gdf = all_df.groupby(["cin_week", "season_holidayed_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_week","season_holidayed_code", "cin_week_seas_count"]
all_df = all_df.merge(gdf, on=["cin_week","season_holidayed_code"], how="left")

gdf = all_df.groupby(["cin_week", "cluster_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_week","cluster_code", "cin_week_clust_count"]
all_df = all_df.merge(gdf, on=["cin_week","cluster_code"], how="left")
###
gdf = all_df.groupby(["cout_mon", "cout_week", "resort_region_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_mon", "cout_week","resort_region_code", "cout_mon_week_resort_reg_count"]
all_df = all_df.merge(gdf, on=["cout_mon", "cout_week","resort_region_code"], how="left")

gdf = all_df.groupby(["cout_mon", "cout_week", "resort_type_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_mon", "cout_week","resort_type_code", "cout_mon_week_resort_typ_count"]
all_df = all_df.merge(gdf, on=["cout_mon", "cout_week","resort_type_code"], how="left")

gdf = all_df.groupby(["cout_mon", "cout_week", "season_holidayed_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_mon", "cout_week","season_holidayed_code", "cout_mon_week_seas_count"]
all_df = all_df.merge(gdf, on=["cout_mon", "cout_week","season_holidayed_code"], how="left")

gdf = all_df.groupby(["cout_mon", "cout_week", "cluster_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_mon", "cout_week","cluster_code", "cout_mon_week_clust_count"]
all_df = all_df.merge(gdf, on=["cout_mon", "cout_week","cluster_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_week", "resort_region_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_mon", "cin_week","resort_region_code", "cin_mon_week_resort_reg_count"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_week","resort_region_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_week", "resort_type_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_mon", "cin_week","resort_type_code", "cin_mon_week_resort_typ_count"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_week","resort_type_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_week", "season_holidayed_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_mon", "cin_week","season_holidayed_code", "cin_mon_week_seas_count"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_week","season_holidayed_code"], how="left")

gdf = all_df.groupby(["cin_mon", "cin_week", "cluster_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_mon", "cin_week","cluster_code", "cin_mon_week_clust_count"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_week","cluster_code"], how="left")
###
gdf = all_df.groupby(["cout_day", "resort_region_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_day","resort_region_code", "cout_day_resort_reg_count"]
all_df = all_df.merge(gdf, on=["cout_day","resort_region_code"], how="left")

gdf = all_df.groupby(["cout_day", "resort_type_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_day","resort_type_code", "cout_day_resort_typ_count"]
all_df = all_df.merge(gdf, on=["cout_day","resort_type_code"], how="left")

gdf = all_df.groupby(["cout_day", "season_holidayed_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_day","season_holidayed_code", "cout_day_seas_count"]
all_df = all_df.merge(gdf, on=["cout_day","season_holidayed_code"], how="left")

gdf = all_df.groupby(["cout_day", "cluster_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_day","cluster_code", "cout_day_clust_count"]
all_df = all_df.merge(gdf, on=["cout_day","cluster_code"], how="left")

gdf = all_df.groupby(["cin_day", "resort_region_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_day","resort_region_code", "cin_day_resort_reg_count"]
all_df = all_df.merge(gdf, on=["cin_day","resort_region_code"], how="left")

gdf = all_df.groupby(["cin_day", "resort_type_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_day","resort_type_code", "cin_day_resort_typ_count"]
all_df = all_df.merge(gdf, on=["cin_day","resort_type_code"], how="left")

gdf = all_df.groupby(["cin_day", "season_holidayed_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_day","season_holidayed_code", "cin_day_seas_count"]
all_df = all_df.merge(gdf, on=["cin_day","season_holidayed_code"], how="left")

gdf = all_df.groupby(["cin_day", "cluster_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_day","cluster_code", "cin_day_clust_count"]
all_df = all_df.merge(gdf, on=["cin_day","cluster_code"], how="left")

###
gdf = all_df.groupby(["cout_week", "cout_day", "resort_region_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_week", "cout_day","resort_region_code", "cout_week_day_resort_reg_count"]
all_df = all_df.merge(gdf, on=["cout_week", "cout_day","resort_region_code"], how="left")

gdf = all_df.groupby(["cout_week", "cout_day", "resort_type_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_week", "cout_day","resort_type_code", "cout_week_day_resort_typ_count"]
all_df = all_df.merge(gdf, on=["cout_week", "cout_day","resort_type_code"], how="left")

gdf = all_df.groupby(["cout_week", "cout_day", "season_holidayed_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_week", "cout_day","season_holidayed_code", "cout_week_day_seas_count"]
all_df = all_df.merge(gdf, on=["cout_week", "cout_day","season_holidayed_code"], how="left")

gdf = all_df.groupby(["cout_week", "cout_day", "cluster_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cout_week", "cout_day","cluster_code", "cout_week_day_clust_count"]
all_df = all_df.merge(gdf, on=["cout_week", "cout_day","cluster_code"], how="left")

gdf = all_df.groupby(["cin_week", "cin_day", "resort_region_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_week", "cin_day","resort_region_code", "cin_week_day_resort_reg_count"]
all_df = all_df.merge(gdf, on=["cin_week", "cin_day","resort_region_code"], how="left")

gdf = all_df.groupby(["cin_week", "cin_day", "resort_type_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_week", "cin_day","resort_type_code", "cin_week_day_resort_typ_count"]
all_df = all_df.merge(gdf, on=["cin_week", "cin_day","resort_type_code"], how="left")

gdf = all_df.groupby(["cin_week", "cin_day", "season_holidayed_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_week", "cin_day","season_holidayed_code", "cin_week_day_seas_count"]
all_df = all_df.merge(gdf, on=["cin_week", "cin_day","season_holidayed_code"], how="left")

gdf = all_df.groupby(["cin_week", "cin_day", "cluster_code"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_week", "cin_day","cluster_code", "cin_week_day_clust_count"]
all_df = all_df.merge(gdf, on=["cin_week", "cin_day","cluster_code"], how="left")

##Number of reservations made w.r.t various resort based combinations of categories
gdf = all_df.groupby(["resort_region_code", "resort_type_code", "state_code_resort", "cluster_code", "reservationstatusid_code", "resort_id"])["reservation_id"].count().reset_index()
gdf.columns = ["resort_region_code", "resort_type_code", "state_code_resort", "cluster_code", "reservationstatusid_code", "resort_id", "resort_re_ty_st_cl_rese_id_count"]
all_df = all_df.merge(gdf, on=["resort_region_code", "resort_type_code", "state_code_resort", "cluster_code", "reservationstatusid_code", "resort_id"], how="left")

##Number of reservations made w.r.t various user based combinations of categories
gdf = all_df.groupby(["main_product_code", "persontravellingid", "state_code_residence", "member_age_buckets"])["reservation_id"].count().reset_index()
gdf.columns = ["main_product_code", "persontravellingid", "state_code_residence", "member_age_buckets", "mem_pro_pers_st_age_count"]
all_df = all_df.merge(gdf, on=["main_product_code", "persontravellingid", "state_code_residence", "member_age_buckets"], how="left")

##Number of reservations made w.r.t checkin_month_day and various resort based combinations of categories
gdf = all_df.groupby(["cin_mon", "cin_day", "resort_region_code", "resort_type_code", "state_code_resort", "cluster_code", "reservationstatusid_code", "resort_id"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_mon", "cin_day", "resort_region_code", "resort_type_code", "state_code_resort", "cluster_code", "reservationstatusid_code", "resort_id", "resort_cin_mon_day_re_ty_st_cl_rese_id_count"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day", "resort_region_code", "resort_type_code", "state_code_resort", "cluster_code", "reservationstatusid_code", "resort_id"], how="left")

##Number of reservations made w.r.t checkin_month_day and various user based combinations of categories
gdf = all_df.groupby(["cin_mon", "cin_day","main_product_code", "persontravellingid", "state_code_residence", "member_age_buckets"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_mon", "cin_day","main_product_code", "persontravellingid", "state_code_residence", "member_age_buckets", "mem_cin_mon_day_pro_pers_st_age_count"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day","main_product_code", "persontravellingid", "state_code_residence", "member_age_buckets"], how="left")

##Number of reservations made w.r.t checkin_month_day and various member count based combinations of categories
gdf = all_df.groupby(["cin_mon", "cin_day","channel_code", "numberofadults", "numberofchildren", "roomnights", "total_pax", "member_age_buckets"])["reservation_id"].count().reset_index()
gdf.columns = ["cin_mon", "cin_day","channel_code", "numberofadults", "numberofchildren", "roomnights", "total_pax", "member_age_buckets", "cin_mon_day_chan_adu_chil_night_pax_age_count"]
all_df = all_df.merge(gdf, on=["cin_mon", "cin_day","channel_code", "numberofadults", "numberofchildren", "roomnights", "total_pax", "member_age_buckets"], how="left")

###Number of reservations made w.r.t booking_month_day and various resort based combinations of categories
gdf = all_df.groupby(["b_mon", "b_day", "resort_region_code", "resort_type_code", "state_code_resort", "cluster_code", "reservationstatusid_code", "resort_id"])["reservation_id"].count().reset_index()
gdf.columns = ["b_mon", "b_day", "resort_region_code", "resort_type_code", "state_code_resort", "cluster_code", "reservationstatusid_code", "resort_id", "resort_b_mon_day_re_ty_st_cl_rese_id_count"]
all_df = all_df.merge(gdf, on=["b_mon", "b_day", "resort_region_code", "resort_type_code", "state_code_resort", "cluster_code", "reservationstatusid_code", "resort_id"], how="left")

###Number of reservations made w.r.t booking_month_day and various user based combinations of categories
gdf = all_df.groupby(["b_mon", "b_day","main_product_code", "persontravellingid", "state_code_residence", "member_age_buckets"])["reservation_id"].count().reset_index()
gdf.columns = ["b_mon", "b_day","main_product_code", "persontravellingid", "state_code_residence", "member_age_buckets", "mem_b_mon_day_pro_pers_st_age_count"]
all_df = all_df.merge(gdf, on=["b_mon", "b_day","main_product_code", "persontravellingid", "state_code_residence", "member_age_buckets"], how="left")

###Number of reservations made w.r.t booking_month_day and various member count based combinations of categories
gdf = all_df.groupby(["b_mon", "b_day","channel_code", "numberofadults", "numberofchildren", "roomnights", "total_pax", "member_age_buckets"])["reservation_id"].count().reset_index()
gdf.columns = ["b_mon", "b_day","channel_code", "numberofadults", "numberofchildren", "roomnights", "total_pax", "member_age_buckets", "b_mon_day_chan_adu_chil_night_pax_age_count"]
all_df = all_df.merge(gdf, on=["b_mon", "b_day","channel_code", "numberofadults", "numberofchildren", "roomnights", "total_pax", "member_age_buckets"], how="left")

#Descriptive stats of various derived numeric intutive features w.r.t checkin_month_day and Number of roomnights booked,
gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["diff_cin_b"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_diff_cin_b_min", "cin_day_mon_nights_diff_cin_b_max", "cin_day_mon_nights_diff_cin_b_mean", "cin_day_mon_nights_diff_cin_b_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["diff_cout_cin"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_diff_cout_cin_min", "cin_day_mon_nights_diff_cout_cin_max", "cin_day_mon_nights_diff_cout_cin_mean", "cin_day_mon_nights_diff_cout_cin_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["t_mem"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_t_mem_min", "cin_day_mon_nights_t_mem_max", "cin_day_mon_nights_t_mem_mean", "cin_day_mon_nights_t_mem_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["r_child_adult"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_r_child_adult_min", "cin_day_mon_nights_r_child_adult_max", "cin_day_mon_nights_r_child_adult_mean", "cin_day_mon_nights_r_child_adult_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["r_room_night"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_r_room_night_min", "cin_day_mon_nights_r_room_night_max", "cin_day_mon_nights_r_room_night_mean", "cin_day_mon_nights_r_room_night_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["r_pax_room_night"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_r_pax_room_night_min", "cin_day_mon_nights_r_pax_room_night_max", "cin_day_mon_nights_r_pax_room_night_mean", "cin_day_mon_nights_r_pax_room_night_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["r_t_mem_room_night"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_r_t_mem_room_night_min", "cin_day_mon_nights_r_t_mem_room_night_max", "cin_day_mon_nights_r_t_mem_room_night_mean", "cin_day_mon_nights_r_t_mem_room_night_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["r_child_room_night"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_r_child_room_night_min", "cin_day_mon_nights_r_child_room_night_max", "cin_day_mon_nights_r_child_room_night_mean", "cin_day_mon_nights_r_child_room_night_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["r_adult_room_night"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_r_adult_room_night_min", "cin_day_mon_nights_r_adult_room_night_max", "cin_day_mon_nights_r_adult_room_night_mean", "cin_day_mon_nights_r_adult_room_night_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["diff1_cin"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_diff1_cin_min", "cin_day_mon_nights_diff1_cin_max", "cin_day_mon_nights_diff1_cin_mean", "cin_day_mon_nights_diff1_cin_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["cin_day", "cin_mon", "roomnights"])["diff2_cin"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["cin_day", "cin_mon","roomnights", "cin_day_mon_nights_diff2_cin_min", "cin_day_mon_nights_diff2_cin_max", "cin_day_mon_nights_diff2_cin_mean", "cin_day_mon_nights_diff2_cin_std"]
all_df = all_df.merge(gdf, on=["cin_day", "cin_mon", "roomnights"], how="left")

gdf = all_df.groupby(["b_day", "b_mon", "roomnights"])["diff1_b"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["b_day", "b_mon","roomnights", "b_day_mon_nights_diff1_b_min", "b_day_mon_nights_diff1_b_max", "b_day_mon_nights_diff1_b_mean", "b_day_mon_nights_diff1_b_std"]
all_df = all_df.merge(gdf, on=["b_day", "b_mon", "roomnights"], how="left")

gdf = all_df.groupby(["b_day", "b_mon", "roomnights"])["diff2_b"].agg(["min", "max", "mean", "std"]).reset_index()
gdf.columns = ["b_day", "b_mon","roomnights", "b_day_mon_nights_diff2_b_min", "b_day_mon_nights_diff2_b_max", "b_day_mon_nights_diff2_b_mean", "b_day_mon_nights_diff2_b_std"]
all_df = all_df.merge(gdf, on=["b_day", "b_mon", "roomnights"], how="left")

#Sorting data
all_df = all_df.sort_values(by="checkin_date_ord").reset_index(drop=True)

##List out all the new features
dtypes = all_df.dtypes.to_frame('dtypes').reset_index()

#Split Train and Test
train_df = all_df[all_df["train_set"]==1].reset_index(drop=True)
test_df = all_df[all_df["train_set"]==0].reset_index(drop=True)

#Preparing data for model building
cols_to_leave = ["reservation_id", "booking_date", "checkin_date", "checkout_date", "amount_spent_per_room_night_scaled", "train_set","checkin_date_ord", "booking_date_ord", "memberid"]
test_df = test_df.drop(['amount_spent_per_room_night_scaled'],axis=1)
fe_col = [col for col in train_df.columns if col not in ID_y_col]
cols_to_use = []
cols_to_use = [i for i in fe_col if i not in cols_to_leave]
train_X = train_df[cols_to_use]
test_X = test_df[cols_to_use]
train_y = (train_df["amount_spent_per_room_night_scaled"]).values
test_id = test_df["reservation_id"].values

#LightGBM function to define hyperparameters, early stopping; to get feature importance, loss
def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed=0, dep=8, rounds=30000):
	params = {}
	params["objective"] = "regression"
	params['metric'] = 'rmse'
	params["max_depth"] = dep
	params["min_data_in_leaf"] =150
	params["learning_rate"] = 0.01
#	params["min_gain_to_split"] = 0.5
	params["bagging_fraction"] = 0.7
	params["feature_fraction"] = 0.7
	params["bagging_freq"] = 5
	params["bagging_seed"] = seed
#	params["scale_pos_weight"] = 3.5
#	params["is_unbalance"] = True
	params["min_sum_hessian_in_leaf"] = 0
	params["lambda_l1"] = 1.5
	params["lambda_l2"] = 0.5
	params["num_leaves"] = 15
#	params["max_bin"] = 350
	params["verbosity"] = 0
	num_rounds = rounds

	plst = list(params.items())
	lgtrain = lgb.Dataset(train_X, label=train_y)

	if test_y is not None:
		lgtest = lgb.Dataset(test_X, label=test_y)
		model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtrain,lgtest], early_stopping_rounds=400, verbose_eval=1000)
	else:
		lgtest = lgb.Dataset(test_X)
		model = lgb.train(params, lgtrain, num_rounds)

	pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
	if test_X2 is not None:
		pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
	print("Features importance...")
	gain = model.feature_importance('gain')
	ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
	ft.to_csv("cm_fimp_2.2.csv", index=False)
	print(ft.head(25))

	loss = 0
	if test_y is not None:
		loss = np.sqrt(metrics.mean_squared_error(test_y, pred_test_y))
		print (loss)
		return pred_test_y, loss, pred_test_y2, model.best_iteration
	else:
		return pred_test_y

#cross validation strategy, used memberid based kfold
#Train and Test data have different set of members ids, cv is defined in such a way Dev/Val data have different set of memberid to mimic train/test
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
train_unique_member = np.array(train_df["memberid"].unique())

#Model Building
print ("Model building..")
model_name = "LGB"
cv_scores = []
pred_test_full = 0
pred_val_full = np.zeros(train_df.shape[0])
for dev_index, val_index in kf.split(train_unique_member):
	dev_mem, val_mem = train_unique_member[dev_index].tolist(), train_unique_member[val_index].tolist()
	dev_X, val_X = train_X[train_df['memberid'].isin(dev_mem)], train_X[train_df['memberid'].isin(val_mem)]
	dev_y, val_y = train_y[train_df['memberid'].isin(dev_mem)], train_y[train_df['memberid'].isin(val_mem)]
	print (dev_X.shape, val_X.shape)

	pred_val, loss, pred_test, nrounds = runLGB(dev_X, dev_y, val_X, val_y, test_X)
	pred_test = runLGB(train_X, train_y, test_X, rounds=nrounds, seed=2018)

	pred_test_full += pred_test
	pred_val_full[train_df['memberid'].isin(val_mem)] = pred_val
	loss = np.sqrt(metrics.mean_squared_error(train_y[train_df['memberid'].isin(val_mem)], pred_val))
	cv_scores.append(loss)
	print(cv_scores)
print(np.mean(cv_scores))
pred_test_full /= 10.
print (np.sqrt(metrics.mean_squared_error(train_y, pred_val_full)))


#Submissions
out_df = pd.DataFrame({"reservation_id":test_id})
out_df["amount_spent_per_room_night_scaled"] = pred_test_full
out_df.to_csv("cm_2.2_test.csv", index=False)