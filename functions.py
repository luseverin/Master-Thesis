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


def comp_rsq(model,yobs):
    '''Function that compute Rsquared (explained variance) taking the observed response 
    variable and the fitted response as input'''
    ybar = np.mean(yobs)
    yfit = model.fittedvalues
    Rsquared = 1 - np.sum((yfit-yobs)**2)/np.sum((yobs-ybar)**2)
    return Rsquared

def comp_adj_rsq(model,yobs,X):
    '''Function that compute the adjusted Rsquared (adjusted explained variance) 
    taking the observed response variable, the fitted response , and the matrix of the predictors as input'''
    npreds = X.shape[1]
    rsq = comp_rsq(model,yobs)
    adj_rsq = 1 - (1-rsq)*(len(yobs)-1)/(len(yobs)-npreds)
    return adj_rsq

def make_fn(addlist,basename="",sep="_",filetype=''):
    return sep.join(addlist)+sep+basename+filetype

def get_lat_lon_res(ds):
    '''Function to obtain the average lat and lon gridspacing from a dataset of a non regular model grid. '''
    lat = ds.coords['lat']
    lon = ds.coords['lon']
    difflat = lat - lat.shift(lat=1)
    latres = difflat.mean().to_numpy()
    difflon = lon - lon.shift(lon=1)
    lonres = difflon.mean().to_numpy()
    return latres, lonres

def mask_qt(ds,q,mask_abs=None,timeres='day',stack=True,pastname='historical',futname='ssp585',cutarea=1000000):
    '''Function taking a dateset as an input, and returning it with the values below the quantile q at each grid cell
       masked. The mask is computed for each gridcell for the past period. Fields for which less than cutarea is above the
       quantile are dropped'''
    
    Upast = ds[pastname]
    Ufut = ds[futname]
    if stack:
        Upast = Upast.stack(real=("member",timeres))
        Ufut = Ufut.stack(real=("member",timeres))
        dim="real"  
    else:
        dim = timeres
    
    Upast_qt = Upast.quantile(q,dim=dim)
    U_mask_past = Upast.where(Upast>Upast_qt)
    U_mask_fut = Ufut.where(Ufut>Upast_qt)
    
    #mask values below threshold
    if mask_abs:
        U_mask_past = U_mask_past.where(U_mask_past>=mask_abs)
        U_mask_fut = U_mask_fut.where(U_mask_fut>=mask_abs)
        
    latres, lonres = get_lat_lon_res(ds)
    gcarea = latres*lonres*100*100 #gridcell area approximated: 1 deg corresponds to 100km
    threshold = round(cutarea/gcarea)
    
    U_mask_past = U_mask_past.dropna(dim=dim,thresh=threshold) #keep fields for which at least X values are not NaN
    U_mask_fut = U_mask_fut.dropna(dim=dim,thresh=threshold) #keep fields for which at least X values are not NaN
    
    #unstack and assemble
    U_mask_past = U_mask_past.unstack().fillna(0)
    U_mask_fut = U_mask_fut.unstack().fillna(0)
    if futname != pastname:
        U_mask_fut.name = futname
        U_mask = xr.combine_by_coords([U_mask_past, U_mask_fut]).fillna(0)
    else:
        U_mask = U_mask_past
   
    return U_mask

def exc_mask_qt(ds,q,timeres='day',stack=True,pastname='historical',futname='ssp585',cutarea=1000000):
    '''Function taking a dateset as an input, and returning it with the values below the quantile q at each grid cell
       masked. The mask is computed for each gridcell for the past period. Fields for which less than cutarea is above the
       quantile are dropped'''
    
    Upast = ds[pastname]
    Ufut = ds[futname]
    if stack:
        Upast = Upast.stack(real=("member",timeres))
        Ufut = Ufut.stack(real=("member",timeres))
        dim="real"  
    else:
        dim = timeres
    
    Upast_qt = Upast.quantile(q,dim=dim)
    U_mask_past = Upast.where(Upast>Upast_qt)
    U_mask_fut = Ufut.where(Ufut>Upast_qt)
    latres, lonres = get_lat_lon_res(ds)
    gcarea = latres*lonres*100*100 #gridcell area approximated: 1 deg corresponds to 100km
    threshold = round(cutarea/gcarea)
    
    U_mask_past_red = U_mask_past.dropna(dim=dim,thresh=threshold) #keep fields for which at least X values are not NaN
    U_mask_fut_red = U_mask_fut.dropna(dim=dim,thresh=threshold) #keep fields for which at least X values are not NaN
    
    #unstack and assemble
    U_mask_past_red = U_mask_past_red.unstack().fillna(0)
    U_mask_fut_red = U_mask_fut_red.unstack().fillna(0)
    
    #take remaining days
    U_mask_past = U_mask_past.unstack().drop_sel(day=U_mask_past_red.day).fillna(0)
    U_mask_fut = U_mask_fut.unstack().drop_sel(day=U_mask_fut_red.day).fillna(0)
    
    if futname != pastname:
        U_mask_fut.name = futname
        U_mask = xr.combine_by_coords([U_mask_past, U_mask_fut]).fillna(0)
    else:
        U_mask = U_mask_past
   
    return U_mask

def diff_qt(ds,q,mask_abs=None,timeres='day',stack=True,pastname='historical',futname='ssp585',cutarea=1000000):
    '''Same as mask_qt but scaling by the q quantile.'''
    Upast = ds[pastname]
    Ufut = ds[futname]
    
    if stack:
        Upast = Upast.stack(real=("member",timeres))
        Ufut = Ufut.stack(real=("member",timeres))
        dim = "real"
    else:
        dim = timeres
        
    Upast_qt = Upast.quantile(q,dim=dim)
    U_mask_past = Upast.where(Upast>Upast_qt)
    U_mask_fut = Ufut.where(Ufut>Upast_qt)
    
    if mask_abs:
        U_mask_past = U_mask_past.where(U_mask_past>=mask_abs)
        U_mask_fut = U_mask_fut.where(U_mask_fut>=mask_abs)
        
    U_diff_past = (U_mask_past-Upast_qt)
    U_diff_fut = (U_mask_fut-Upast_qt)

    latres, lonres = get_lat_lon_res(ds)
    gcarea = latres*lonres*100*100 #gridcell area approximated: 1 deg corresponds to 100km
    threshold = round(cutarea/gcarea)
    
    U_diff_past = U_diff_past.dropna(dim=dim,thresh=threshold) #keep fields for which at least X values are not NaN
    U_diff_fut = U_diff_fut.dropna(dim=dim,thresh=threshold) #keep fields for which at least X values are not NaN
    
    #unstack and assemble
    U_diff_past = U_diff_past.unstack().fillna(0)
    U_diff_fut = U_diff_fut.unstack().fillna(0)
    if futname != pastname:
        U_diff_fut.name = futname
        U_diff = xr.combine_by_coords([U_diff_past, U_diff_fut]).fillna(0)
    else:
        U_diff = U_diff_past
    return U_diff

def exc_diff_qt(ds,q,timeres='day',stack=True,pastname='historical',futname='ssp585',cutarea=1000000):
    '''Same as mask_qt but scaling by the q quantile.'''
    Upast = ds[pastname]
    Ufut = ds[futname]
    
    if stack:
        Upast = Upast.stack(real=("member",timeres))
        Ufut = Ufut.stack(real=("member",timeres))
        dim = "real"
    else:
        dim = timeres
        
    Upast_qt = Upast.quantile(q,dim=dim)
    U_mask_past = Upast.where(Upast>Upast_qt)
    U_mask_fut = Ufut.where(Ufut>Upast_qt)
    U_diff_past = (U_mask_past-Upast_qt)
    U_diff_fut = (U_mask_fut-Upast_qt)

    latres, lonres = get_lat_lon_res(ds)
    gcarea = latres*lonres*100*100 #gridcell area approximated: 1 deg corresponds to 100km
    threshold = round(cutarea/gcarea)
    print('cell area: '+str(gcarea)+'\nthreshold: '+str(threshold))

    U_diff_past_red = U_diff_past.dropna(dim=dim,thresh=threshold) #keep fields for which at least X values are not NaN
    U_diff_fut_red = U_diff_fut.dropna(dim=dim,thresh=threshold) #keep fields for which at least X values are not NaN
    
    #unstack and assemble
    U_diff_past_red = U_diff_past_red.unstack().fillna(0)
    U_diff_fut_red = U_diff_fut_red.unstack().fillna(0)
    
    #take remaining days
    U_diff_past = U_diff_past.unstack().drop_sel(day=U_diff_past_red.day).fillna(0)
    U_diff_fut = U_diff_fut.unstack().drop_sel(day=U_diff_fut_red.day).fillna(0)
    
    if futname != pastname:
        U_diff_fut.name = futname
        U_diff = xr.combine_by_coords([U_diff_past, U_diff_fut]).fillna(0)
    else:
        U_diff = U_diff_past
    return U_diff

def scale_qt(ds,q,timeres='day',stack=True,pastname='historical',futname='ssp585',cutarea=1000000):
    '''Same as mask_qt but scaling by the q quantile.'''
    Upast = ds[pastname]
    Ufut = ds[futname]
    
    if stack:
        Upast = Upast.stack(real=("member",timeres))
        Ufut = Ufut.stack(real=("member",timeres))
        dim = "real"
    else:
        dim = timeres
    Upast_qt = Upast.quantile(q,dim=dim)
    U_mask_past = Upast.where(Upast>Upast_qt)
    U_mask_fut = Ufut.where(Ufut>Upast_qt)
    U_scaled_past = (U_mask_past-Upast_qt)/Upast_qt
    U_scaled_fut = (U_mask_fut-Upast_qt)/Upast_qt

    latres, lonres = get_lat_lon_res(ds)
    gcarea = latres*lonres*100*100 #gridcell area approximated: 1 deg corresponds to 100km
    threshold = round(cutarea/gcarea)
    
    U_scaled_past = U_scaled_past.dropna(dim=dim,thresh=threshold) #keep fields for which at least X values are not NaN
    U_scaled_fut = U_scaled_fut.dropna(dim=dim,thresh=threshold) #keep fields for which at least X values are not NaN
    
    #unstack and assemble
    U_scaled_past = U_scaled_past.unstack().fillna(0)
    U_scaled_fut = U_scaled_fut.unstack().fillna(0)
    if futname != pastname:
        U_scaled_fut.name = futname
        U_scaled = xr.combine_by_coords([U_scaled_past, U_scaled_fut]).fillna(0)
    else:
        U_scaled = U_scaled_past
    return U_scaled

def exc_scale_qt(ds,q,timeres='day',stack=True,pastname='historical',futname='ssp585',cutarea=1000000):
    '''Same as mask_qt but scaling by the q quantile.'''
    Upast = ds[pastname]
    Ufut = ds[futname]
    
    if stack:
        Upast = Upast.stack(real=("member",timeres))
        Ufut = Ufut.stack(real=("member",timeres))
        dim = "real"
    else:
        dim = timeres
    Upast_qt = Upast.quantile(q,dim=dim)
    U_mask_past = Upast.where(Upast>Upast_qt)
    U_mask_fut = Ufut.where(Ufut>Upast_qt)
    U_scaled_past = (U_mask_past-Upast_qt)/Upast_qt
    U_scaled_fut = (U_mask_fut-Upast_qt)/Upast_qt

    latres, lonres = get_lat_lon_res(ds)
    gcarea = latres*lonres*100*100 #gridcell area approximated: 1 deg corresponds to 100km
    threshold = round(cutarea/gcarea)
    print('cell area: '+str(gcarea)+'\nthreshold: '+str(threshold))
    
    U_scaled_past_red = U_scaled_past.dropna(dim=dim,thresh=threshold) #keep fields for which at least X values are not NaN
    U_scaled_fut_red = U_scaled_fut.dropna(dim=dim,thresh=threshold) #keep fields for which at least X values are not NaN
    
    #unstack and assemble
    U_scaled_past_red = U_scaled_past_red.unstack().fillna(0)
    U_scaled_fut_red = U_scaled_fut_red.unstack().fillna(0)
    
    #take remaining days
    U_scaled_past = U_scaled_past.unstack().drop_sel(day=U_scaled_past_red.day).fillna(0)
    U_scaled_fut = U_scaled_fut.unstack().drop_sel(day=U_scaled_fut_red.day).fillna(0)
    
    if futname != pastname:
        U_scaled_fut.name = futname
        U_scaled = xr.combine_by_coords([U_scaled_past, U_scaled_fut]).fillna(0)
    else:
        U_scaled = U_scaled_past
    return U_scaled


def evnamestr(arrayel):
        '''Transform multiindex (tuple) of event index to a string '''
        imem = arrayel[0]
        iday = arrayel[1]
        strout = "Member: "+str(imem)+", day id: "+str(iday)
        return strout
evnamestr = np.vectorize(evnamestr)

def def_domain(ncdf,min_lat,max_lat,min_lon,max_lon):
    LatIndexer, LonIndexer = 'lat', 'lon'
    ncdf = ncdf.loc[{LatIndexer: slice(min_lat, max_lat),
                      LonIndexer: slice(min_lon, max_lon)}]
    return ncdf

def norm_lon(ncdf):
    ncdf.coords['lon'] = (ncdf.coords['lon'] + 180) % 360 - 180
    return ncdf.sortby(ncdf.lon)
    
def get_ONDJFM_day(ncdf, months=[1,2,3,10,11,12]):
    """get_ONJDFM function select extended winter months (ONDJMF) from a ncdf file containing future and past simulations
    with both years and dates (days if daily, month if monthly) defined within yrs and dates tuples"""
    if "time" not in ncdf.dims:
        ncdf = ncdf.swap_dims({"day":"time"})   
    return ncdf.isel(time=ncdf.time.dt.month.isin(months))

def norm_wind(ncdfu,ncdfv):
    return np.sqrt(ncdfu**2+ncdfv**2)

pp_func_dic_diff = {'Cubic excess-over-threshold':scale_qt,'Emanuel 2011':diff_qt,'Welker 2021':diff_qt,'Scaled sigmoid': scale_qt, 'Schwierz 2010':diff_qt}
pp_func_dic_mask = {'Cubic excess-over-threshold':scale_qt,'Emanuel 2011':mask_qt,'Welker 2021':mask_qt, 'Scaled sigmoid':scale_qt,'Schwierz 2010':mask_qt}
pp_func_dic_exc = {'Cubic excess-over-threshold':exc_scale_qt,'Emanuel 2011':exc_diff_qt,'Welker 2021':exc_diff_qt,'Scaled sigmoid': scale_qt, 'Schwierz 2010':exc_diff_qt}