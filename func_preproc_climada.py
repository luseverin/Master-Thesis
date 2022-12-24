#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:06:06 2022

@author: lucaseverino
"""
## imports
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import copy as cp
from climada.hazard import Hazard,Centroids
from climada.engine import Impact
from scipy import sparse
from timeit import default_timer as timer 
from constants import *

def make_fn(addlist,basename,sep="_",filetype=''):
    return sep.join(addlist)+sep+basename+filetype

def def_domain(ncdf,min_lat,max_lat,min_lon,max_lon):
    LatIndexer, LonIndexer = 'lat', 'lon'
    ncdf = ncdf.loc[{LatIndexer: slice(min_lat, max_lat),
                      LonIndexer: slice(min_lon, max_lon)}]
    return ncdf

def norm_lon(ncdf):
    ncdf.coords['lon'] = (ncdf.coords['lon'] + 180) % 360 - 180
    return ncdf.sortby(ncdf.lon)

def get_ONDJFM(ncdf,yrs,dates,res='month',monout=[0,1,2,9,10,11]):
    """get_ONJDFM function select extended winter months (ONDJMF) from a ncdf file containing future and past simulations
    with both years and dates (days if daily, month if monthly) defined within yrs and dates tuples"""
    if res not in ncdf.coords.keys():
        raise "ValueError: the resolutions do not match"
    
    if res=="month":
        mon_id = (ncdf["month"]-1) % 12
    
        mask = np.isin(mon_id,monout) 
        return ncdf.sel(month=mask)
    
def get_ONDJFM_day(ncdf, months=[1,2,3,10,11,12]):
    if "time" not in ncdf.dims:
        ncdf = ncdf.swap_dims({"day":"time"})   
    return ncdf.isel(time=ncdf.time.dt.month.isin(months))

def norm_wind(ncdfu,ncdfv):
    return np.sqrt(ncdfu**2+ncdfv**2)


def get_lat_lon_res(ds):
    '''Function to obtain the average lat and lon gridspacing from a dataset of a non regular model grid. '''
    lat = ds.coords['lat']
    lon = ds.coords['lon']
    difflat = lat - lat.shift(lat=1)
    latres = difflat.mean().to_numpy()
    difflon = lon - lon.shift(lon=1)
    lonres = difflon.mean().to_numpy()
    return latres, lonres

def mask_qt(ds,q,timeres='day',pastname='historical',futname='ssp585',cutarea=1000000):
    '''Function taking a dateset as an input, and returning it with the values below the quantile q at each grid cell
       masked. The mask is computed for each gridcell for the past period. Fields for which less than cutarea is above the
       quantile are dropped'''
    U = ds.stack(real=("member",timeres))
    Upast = U[pastname]
    Upast_qt = Upast.quantile(q,dim="real")
    U_mask = U.where(U>Upast_qt)
    latres, lonres = get_lat_lon_res(ds)
    gcarea = latres*lonres*100*100 #gridcell area approximated: 1 deg corresponds to 100km
    threshold = round(cutarea/gcarea)
    
    U_mask = U_mask.dropna(dim="real",thresh=threshold) #keep fields for which at least X values are not NaN
    return U_mask.unstack().fillna(0)

def scale_qt(ds,q,timeres='day',pastname='historical',futname='ssp585',cutarea=1000000):
    '''Same as mask_qt but scaling by the q quantile.'''
    U = ds.stack(real=("member",timeres))
    Upast = U[pastname]
    Upast_qt = Upast.quantile(q,dim="real")
    U_mask = U.where(U>Upast_qt)
    U_scaled = (U_mask-Upast_qt)/Upast_qt
    latres, lonres = get_lat_lon_res(ds)
    gcarea = latres*lonres*100*100 #gridcell area approximated: 1 deg corresponds to 100km
    threshold = round(cutarea/gcarea)
    U_scaled = U_scaled.dropna(dim="real",thresh=threshold) #keep fields for which at least X values are not NaN
    return U_scaled.unstack().fillna(0)

def set_centroids(da,haztype='WS',timeres="day",plot=False):
    '''Function which takes a xarray.DataArray as an input and returns a climada.hazard object, with centroids corresponding to 
        latitude and longitude of the DataArray.'''
    
    da = da.reindex(lat=list(reversed(da.lat))) #need to invert latitude
    period = da.name
    lat = da.lat
    lon = da.lon
    
    #compute mean grid spacing (irregular model grid) and round it to 1 digit
    latreso, lonreso = get_lat_lon_res(da)
    latres = latreso.round(4)
    lonres = lonreso.round(4)
    
    #get bondaries and round to 1 digit
    min_lat=lat[0].to_numpy().round(4)
    max_lat=lat[-1].to_numpy().round(4)
    min_lon=lon[0].to_numpy().round(4)
    max_lon=lon[-1].to_numpy().round(4)
    
    print('Lat resolution original: '+str(-latreso)+' ,rounded: '+str(-latres)+'\nLon resolution original: '+str(lonreso)+' ,rounded: '+str(lonres))
    
    # number of points
    n_lat = len(da.lat)
    n_lon = len(da.lon)
    
    #stack first members and time resolution, and then latitutde and longitude to obtain the event matrix
    events = da.stack(events=("member",timeres)) #need to be of size n_ev * ncent = nmem*180 * ncent
    evmat = events.stack(cent=("lat","lon"))
    #print('Evmat: '+str(evmat.shape))
    print('lat: '+str(n_lat)+', lon: '+str(n_lon))
    #nmon = len(da[res]) 
    nmem = len(da["member"])
    n_ev = len(evmat["events"])
    
    #intiate Hazard object, using Centroids.from_pix_bounds
    haz = Hazard(haztype)
    left, bottom, right, top = min_lon, min_lat, max_lon, max_lat
    #scale with 1/2 res?
    xf_lat = min_lat - 0.5*latres
    xo_lon = min_lon - 0.5*lonres
    d_lat = latres
    d_lon = lonres
    #haz.centroids = Centroids.from_pnt_bounds((left, bottom, right, top), xyres) # default crs used
    #haz.centroids = Centroids.from_lat_lon(lat,lon)
    haz.centroids = Centroids.from_pix_bounds(xf_lat, xo_lon, d_lat, d_lon, n_lat, n_lon) # default crs used
    #print('Centroids shape: '+ str(haz.centroids.shape)+', size: '+str(haz.centroids.size))
    haz.intensity = sparse.csr_matrix(evmat)
    haz.units = 'm/s'
    ev_id = np.arange(n_ev, dtype=int)
    haz.event_id = ev_id
    #dayid = ["ID: "+str(ev_id[evid])+ "day: "+ str(day) for evid,day in enumerate(nmem*da.time.dt.strftime('%Y-%m-%d').to_series().to_list())]
    ev_names = events.coords["events"].to_numpy().tolist()
    haz.event_name = ev_names
    #haz.date = [721166]
    haz.orig = np.zeros(n_ev, bool)
    haz.frequency = np.ones(n_ev)/90 
    haz.fraction = haz.intensity.copy()
    haz.fraction.data.fill(1)
    haz.centroids.set_meta_to_lat_lon()
    haz.centroids.set_geometry_points()
    haz.check()
    print('Check centroids borders:', haz.centroids.total_bounds)
    if plot:
        haz.centroids.plot()
    
    return haz

def imp_calc(modname,exp,impf_set,pathin,basename,qt=0.98,cut=1000000,if_id=0,timeres='day',pastname='historical',futname='ssp585',
             savename=None,savefiles={"haz":False,"impcsv":False,"impmat":False,"impds":False}):
    '''Function to compute impacts. Does not require already preprocessed fields. Return hazard and impact climada objects.'''  
    #read netcdf
    fn = pathin+modname+'_'+basename+".nc"
    ncdf = xr.open_dataset(fn)
    ncdfw = ncdf[[pastname,futname]]
    #apply gust factor
    gust_ds = gst_fact*ncdfw
    
    #get name of impact_func
    if_name = impf_set.get_func(haz_type=haz_type,fun_id=0).name
    
    #preprocess fields
    preprocess_func = pp_func_dic[if_name]
    gust_pp = preprocess_func(gust_ds,qt,cutarea=cut,timeres=timeres,pastname=pastname,futname=futname)
    gust_pp_past = gust_pp[pastname]
    gust_pp_fut = gust_pp[futname]
    
    #prepare hazards centroids
    haz_past = set_centroids(gust_pp_past,timeres=timeres)
    haz_fut = set_centroids(gust_pp_fut,timeres=timeres)
#    haz_fut.centroids.set_meta_to_lat_lon()
#    haz_fut.centroids.set_geometry_points()
#    haz_past.centroids.set_meta_to_lat_lon()
#    haz_past.centroids.set_geometry_points()
    
    
    # Exposures: rename column and assign id
    exp.gdf.rename(columns={"impf_": "impf_" + haz_type}, inplace=True)
    exp.gdf['impf_' + haz_type] = if_id
    exp.check()
    exp.gdf.head()
    #deepcopy exposure before assigning centroids
    exp_past = cp.deepcopy(exp)
    exp_fut = cp.deepcopy(exp)
    
    #assign centroids
    exp_fut.assign_centroids(haz_fut,distance='euclidian')
    exp_past.assign_centroids(haz_past,distance='euclidian')
    
    
    #compute impacts
    #past
    start_time = timer()
    imp_past = Impact()
    imp_past.calc(exp_past, impf_set, haz_past, save_mat=True) #Do not save the results geographically resolved (only aggregate values)
    time_delta_past = timer() - start_time
    print(time_delta_past)
    #future
    start_time = timer()
    imp_fut = Impact()
    imp_fut.calc(exp_fut, impf_set, haz_fut, save_mat=True)
    time_delta_fut = timer() - start_time
    print(time_delta_fut)

    ##save files
    if savename:
        savenameb = savename
    else:
        savenameb = basename
    
    savename_past = 'hist_'+modname+'_'+savenameb
    savename_fut = 'ssp585_'+modname+'_'+savenameb
    impf_namesht = impf_sht_names[if_name]
    #save hazards
    if savefiles["haz"]:
        haz_past.write_hdf5('results/hazards/'+savename_past+'.h5')
        haz_fut.write_hdf5('results/hazards/'+savename_fut+'.h5')
    #save impacts
    if savefiles["impcsv"]:
        imp_past.write_csv('results/impacts/impact csv'+impf_namesht+"_"+savename_past+'.csv')
        imp_fut.write_csv('results/impacts/impact csv'+impf_namesht+"_"+savename_fut+'.csv')
        
    if savefiles["impmat"]:
        imp_past.write_sparse_csr('results/impacts/impact matrices/'+impf_namesht+"_"+savename_past+'.npz')
        imp_fut.write_sparse_csr('results/impacts/impact matrices/'+impf_namesht+"_"+savename_fut+'.npz')
        
    if savefiles["impds"]:
        imp_past.write_sparse_csr('results/impacts/impact matrices/'+impf_namesht+"_"+savename_past+'.npz')
        imp_fut.write_sparse_csr('results/impacts/impact matrices/'+impf_namesht+"_"+savename_fut+'.npz')

    return haz_past, haz_fut, imp_past, imp_fut

def imp_calc_hist_fut(res_df,res_df2,modname,exp,impf_set,pathin,basename,qt=0.98,cut=1000000,if_id=0,timeres='day',pastname='historical',futname='ssp585',
             savename=None,savefiles={"haz":False,"impcsv":False,"impmat":False,"impds":False}):
    '''Function to compute impacts and compute some statistic, between historical and future periode. Return dataframe res_df containing
        AAI_agg, 45yrs and 90 yrs impacts for each period, model and impact function'''
    #read netcdf
    fn = pathin+modname+'_'+basename+".nc"
    ncdf = xr.open_dataset(fn)
    ncdfw = ncdf[[pastname,futname]]
    #apply gust factor
    gust_ds = gst_fact*ncdfw
    
    #get name of impact_func
    if_name = impf_set.get_func(haz_type=haz_type,fun_id=0).name
    
    #preprocess fields
    preprocess_func = pp_func_dic[if_name]

    gust_pp = preprocess_func(gust_ds,qt,cutarea=cut,timeres=timeres,pastname=pastname,futname=futname)
    gust_pp_past = gust_pp[pastname]
    gust_pp_fut = gust_pp[futname]
    
    #prepare hazards centroids
    haz_past = set_centroids(gust_pp_past,timeres=timeres)
    haz_fut = set_centroids(gust_pp_fut,timeres=timeres)
#    haz_fut.centroids.set_meta_to_lat_lon()
#    haz_fut.centroids.set_geometry_points()
#    haz_past.centroids.set_meta_to_lat_lon()
#    haz_past.centroids.set_geometry_points()
    
    
    # Exposures: rename column and assign id
    exp.gdf.rename(columns={"impf_": "impf_" + haz_type}, inplace=True)
    exp.gdf['impf_' + haz_type] = if_id
    exp.check()
    exp.gdf.head()
    #deepcopy exposure before assigning centroids
    exp_past = cp.deepcopy(exp)
    exp_fut = cp.deepcopy(exp)
    
    #assign centroids
    exp_fut.assign_centroids(haz_fut,distance='euclidian')
    exp_past.assign_centroids(haz_past,distance='euclidian')
    
    
    #compute impacts
    #past
    start_time = timer()
    imp_past = Impact()
    imp_past.calc(exp_past, impf_set, haz_past, save_mat=True) #Do not save the results geographically resolved (only aggregate values)
    time_delta_past = timer() - start_time
    print(time_delta_past)
    #future
    start_time = timer()
    imp_fut = Impact()
    imp_fut.calc(exp_fut, impf_set, haz_fut, save_mat=True)
    time_delta_fut = timer() - start_time
    print(time_delta_fut)
    #compute freq curves 
    imp45_past = imp_past.calc_freq_curve(return_per=45).impact 
    imp45_fut = imp_fut.calc_freq_curve(return_per=45).impact 
    imp90_past = imp_past.calc_freq_curve(return_per=90).impact 
    imp90_fut = imp_fut.calc_freq_curve(return_per=90).impact 
    #save results
    res_df.loc[modname,("AAI_agg","past",if_name)] = imp_past.aai_agg
    res_df.loc[modname,("AAI_agg","future",if_name)] = imp_fut.aai_agg
    res_df.loc[modname,("45 yr impact","past",if_name)] = imp45_past
    res_df.loc[modname,("45 yr impact","future",if_name)] = imp45_fut
    res_df.loc[modname,("90 yr impact","past",if_name)] = imp90_past
    res_df.loc[modname,("90 yr impact","future",if_name)] = imp90_fut
    
    #save results v2
    res_df2.loc[(modname,if_name),("AAI_agg","past")] = imp_past.aai_agg
    res_df2.loc[(modname,if_name),("AAI_agg","future")] = imp_fut.aai_agg
    res_df2.loc[(modname,if_name),("45 yr impact","past")] = imp45_past
    res_df2.loc[(modname,if_name),("45 yr impact","future")] = imp45_fut
    res_df2.loc[(modname,if_name),("90 yr impact","past")] = imp90_past
    res_df2.loc[(modname,if_name),("90 yr impact","future")] = imp90_fut
        
    ##save files
    if savename:
        savenameb = savename
    else:
        savenameb = basename
    
    savename_past = 'hist_'+modname+'_'+savenameb
    savename_fut = 'ssp585_'+modname+'_'+savenameb
    impf_namesht = impf_sht_names[if_name]
    #save hazards
    if savefiles["haz"]:
        haz_past.write_hdf5('results/hazards/'+savename_past+'.h5')
        haz_fut.write_hdf5('results/hazards/'+savename_fut+'.h5')
    #save impacts
    if savefiles["impcsv"]:
        imp_past.write_csv('results/impacts/impact csv'+impf_namesht+"_"+savename_past+'.csv')
        imp_fut.write_csv('results/impacts/impact csv'+impf_namesht+"_"+savename_fut+'.csv')
        
    if savefiles["impmat"]:
        imp_past.write_sparse_csr('results/impacts/impact matrices/'+impf_namesht+"_"+savename_past+'.npz')
        imp_fut.write_sparse_csr('results/impacts/impact matrices/'+impf_namesht+"_"+savename_fut+'.npz')

    return 

def evnamestr(arrayel):
        '''Transform multiindex (tuple) of event index to a string '''
        imem = arrayel[0]
        iday = arrayel[1]
        strout = "Member: "+str(imem)+", day id: "+str(iday)
        return strout
evnamestr = np.vectorize(evnamestr)
