### the functions are adapted from Pu Zheng's ImageAnalysis3

import numpy as np
from scipy.ndimage.filters import maximum_filter,minimum_filter,gaussian_filter
from pyfftw.interfaces.numpy_fft import rfftn,irfftn
from scipy import fftpack
from scipy.spatial import cKDTree as KDTree
from scipy.optimize import leastsq
from scipy.ndimage import binary_dilation

# remove edge points
def remove_edge_points(im, T_seeds, distance=2):
    
    im_size = np.array(np.shape(im))
    _seeds = np.array(T_seeds)[:len(im_size),:].transpose()
    flags = []
    for _seed in _seeds:
        _f = ((_seed >= distance) * (_seed <= im_size-distance)).all()
        flags.append(_f)
    
    return np.array(flags, dtype=bool)

# integrated function to get seeds
def get_seeds(im, max_num_seeds=None, th_seed=1000, 
              gfilt_size=0.75, background_gfilt_size=7.5,
              filt_size=3, min_edge_distance=2,
              use_dynamic_th=True, dynamic_niters=10, min_dynamic_seeds=100,
              minimum_threshold = 500, remove_hot_pixel=True, hot_pixel_th=5,
              segment=None):
    """Function to fully get seeding pixels given a image and thresholds.
    Inputs:
      im: image given, np.ndarray, 
      max_num_seeds: number of max seeds number, int default=-1, 
      th_seed: seeding threshold between max_filter - min_filter, float/int, default=150, 
      use_percentile: whether use percentile to determine seed_th, bool, default=False,
      th_seed_per: seeding percentile in intensities, float/int of percentile, default=95, 
      gfilt_size: gaussian filter size for max_filter image, float, default=0.75, 
      background_gfilt_size: gaussian filter size for min_filter image, float, default=10,
      filt_size: filter size for max/min filter, int, default=3, 
      min_edge_distance: minimal allowed distance for seed to image edges, int/float, default=3,
      use_dynamic_th: whetaher use dynamic th_seed, bool, default=True, 
      dynamic_niters: number of iterations used for dynamic th_seed, int, default=10, 
      min_dynamic_seeds: minimal number of seeds to get with dynamic seeding, int, default=1,
    """
    # check inputs
    if not isinstance(im, np.ndarray):
        raise TypeError(f"image given should be a numpy.ndarray, but {type(im)} is given.")
    _local_edges = np.zeros(len(np.shape(im)))
    # return if there is no cell
    if np.count_nonzero(segment)==0:
        return []
    ## do seeding
    if not use_dynamic_th:
        dynamic_niters = 1 # setting only do seeding once
    else:
        dynamic_niters = int(dynamic_niters)
    # front filter:
    if gfilt_size:
        _max_im = np.array(gaussian_filter(im, gfilt_size), dtype=im.dtype)
    else:
        _max_im = np.array(im, dtype=im.dtype)
    _max_ft = np.array(maximum_filter(_max_im, int(filt_size)) == _max_im, dtype=bool)

    # background filter
    if background_gfilt_size:
        _min_im = np.array(gaussian_filter(im, background_gfilt_size), dtype=im.dtype)
    else:
        _min_im = np.array(im, dtype=im.dtype)
    _min_ft = np.array(minimum_filter(_min_im, int(filt_size)) != _min_im, dtype=bool)
    
    # generate map
    _local_maximum_mask = (_max_ft & _min_ft).astype(bool)
    if segment is not None:
        segment_mask = segment>0
        dilated_mask = [binary_dilation(ly, structure=np.ones((29,29))) for ly in segment_mask]
        dilated_mask = np.array(dilated_mask)
        _local_maximum_mask = _local_maximum_mask*dilated_mask
        del segment_mask
        del dilated_mask
    _diff_ft = (_max_im.astype(float) - _min_im.astype(float))
    
    # clear RAM immediately
    del(_max_im, _min_im)
    del(_max_ft, _min_ft) 

    # iteratively select seeds
    for _iter in range(dynamic_niters):
        # get seed coords
        _current_seed_th = th_seed * (1-_iter/dynamic_niters)
        # end if the threshold is too low
        if _current_seed_th<=minimum_threshold:
            break
        
        # get seeds
        _coords = np.where(_local_maximum_mask & (_diff_ft >= _current_seed_th))
        # remove edges
        if min_edge_distance > 0:
            _keep_flags = remove_edge_points(im, _coords, min_edge_distance)
            _coords = tuple(_cs[_keep_flags] for _cs in _coords)
        
        # if got enough seeds, proceed.
        if len(_coords[0]) >= min_dynamic_seeds:
            break
    # hot pixels
    if remove_hot_pixel:
        _,_x,_y = _coords
        _xy_str = [str([np.round(x_,1),np.round(y_,1)]) 
                    for x_,y_ in zip(_x,_y)]
        _unique_xy_str, _cts = np.unique(_xy_str, return_counts=True)
        _keep_hot = np.array([_xy not in _unique_xy_str[_cts>=hot_pixel_th] 
                             for _xy in _xy_str],dtype=bool)
        _coords = tuple(_cs[_keep_hot] for _cs in _coords)
    # get heights
    _hs = _diff_ft[_coords]
    _final_coords = np.array(_coords) + _local_edges[:, np.newaxis] # adjust to absolute coordinates
    # patch heights
    # transpose and sort by intensity decreasing order
    _final_coords = np.transpose(_final_coords)[np.flipud(np.argsort(_hs))]
   
    # truncate with max_num_seeds
    if max_num_seeds is not None and max_num_seeds > 0 and max_num_seeds <= len(_final_coords):
        _final_coords = _final_coords[:int(max_num_seeds)]
        print(f"--- {max_num_seeds} seeds are kept.")
    
    return _final_coords


class GaussianFit():
    def __init__(self,im,X,center=None,n_aprox=10,min_w=0.5,max_w=4.,delta_center=3.,
                 init_w=1.5):
        self.min_w = min_w*min_w
        self.max_w = max_w*max_w
        
        self.delta_center = delta_center
        self.im = np.array(im,dtype=np.float32)
        self.x,self.y,self.z = np.array(X,dtype=np.float32)
        #get estimates
        argsort_im = np.argsort(im)
        if center is None:
            center = np.median(X[:,argsort_im][:,-n_aprox:],-1)
        self.center_est = center
        sorted_im = im[argsort_im]
        eps =  np.exp(-10.)
        bk_guess = np.log(np.max([np.mean(sorted_im[:n_aprox]),eps]))
        h_guess = np.log(np.max([np.mean(sorted_im[-n_aprox:]),eps]))
        wsq = init_w**2
        wg = np.log((self.max_w - wsq)/(wsq-self.min_w))
        self.p_ = np.array([bk_guess,h_guess,0,0,0,wg,wg,wg,0,0],dtype=np.float32)
        self.to_natural_paramaters()
        self.success = False

    def to_center(self,c0_,c1_,c2_):
        """constrains via sigmoidal function close to local center"""
        delta = self.delta_center
        # c0
        if c0_ >= np.log(np.finfo(c0_.dtype).max):
            c0 = - delta + self.center_est[0]
        elif c0_ <= - np.log(np.finfo(c0_.dtype).max):
            c0 = delta + self.center_est[0]
        else:
            c0 = 2. *delta / (1.+np.exp(c0_)) - delta + self.center_est[0]
        # c1
        if c1_ >= np.log(np.finfo(c1_.dtype).max):
            c1 = - delta + self.center_est[1]
        elif c1_ <= - np.log(np.finfo(c1_.dtype).max):
            c1 = delta + self.center_est[1]
        else:
            c1 = 2. *delta / (1.+np.exp(c1_)) - delta + self.center_est[1]
        # c2
        if c2_ >= np.log(np.finfo(c2_.dtype).max):
            c2 = - delta + self.center_est[2]
        elif c2_ <= - np.log(np.finfo(c2_.dtype).max):
            c2 = delta + self.center_est[2]
        else:
            c2 = 2. *delta / (1.+np.exp(c2_)) - delta + self.center_est[2]

        #c0 = 2. *delta / (1.+np.exp(c0_)) - delta + self.center_est[0]
        #c1 = 2.*delta/(1.+np.exp(c1_))-delta+self.center_est[1]
        #c2 = 2.*delta/(1.+np.exp(c2_))-delta+self.center_est[2]
        return c0,c1,c2

    def to_sine(self, t_):
        """constrain sin-angles to -1,1"""
        #eps =  10E-5
        #self.sine_eps = eps
        #return 2.*(1-eps)/(1+np.exp(t_))-1.+eps
        if t_ >= np.log(np.finfo(t_.dtype).max):
            return -1
        elif t_ <= - np.log(np.finfo(t_.dtype).max):
            return 1
        else:
            return 2./(1+np.exp(t_))-1.

    def to_ws(self,w_):
        """constrain widths"""
        min_ws = self.min_w
        delta_ws = self.max_w - min_ws

        if w_ >= np.log(np.finfo(w_.dtype).max):
            ws = min_ws
        elif w_ <= - np.log(np.finfo(w_.dtype).max):
            ws = delta_ws + min_ws
        else:
            ws = delta_ws/(1.+np.exp(w_))+min_ws
        return ws

    def to_natural_paramaters(self,parms=None):
        """
        Convert from constrained paramaters to [hf,xc,yc,zc,bkf,w1f,w2f,w3f,t,p,eps]
        """
        if parms is None:
            parms = self.p_
        bk,h,xp,yp,zp,w1,w2,w3,pp,tp = parms
        bkf,hf=np.exp(bk),np.exp(h)
        t,p = self.to_sine(tp),self.to_sine(pp)
        w1f,w2f,w3f = np.sqrt(self.to_ws(w1)),np.sqrt(self.to_ws(w2)),np.sqrt(self.to_ws(w3))
        xc,yc,zc = self.to_center(xp,yp,zp)
        eps = self.calc_eps(parms)
        eps = np.mean(np.abs(eps))
        self.p = np.array([hf,xc,yc,zc,bkf,w1f,w2f,w3f,t,p,eps],dtype=np.float32)
        return self.p
    def calc_f(self,parms):
        self.p_ = parms
        
        bk,h,xp,yp,zp,w1,w2,w3,pp,tp = parms
        t,p = self.to_sine(tp),self.to_sine(pp)
        ws1,ws2,ws3 = self.to_ws(w1),self.to_ws(w2),self.to_ws(w3)
        xc,yc,zc = self.to_center(xp,yp,zp)
        xt,yt,zt = self.x-xc,self.y-yc,self.z-zc
        
        p2 = p*p
        t2 = t*t
        tc2 = 1-t2
        pc2 = 1-p2
        tc= np.sqrt(tc2)
        pc= np.sqrt(pc2)
        s1,s2,s3 = 1./ws1,1./ws2,1./ws3
        x2c = pc2*tc2*s1 + t2*s2 + p2*tc2*s3
        y2c = pc2*t2*s1 + tc2*s2 + p2*t2*s3
        z2c = p2*s1 + pc2*s3
        xyc = 2*tc*t*(pc2*s1 - s2 + p2*s3)
        xzc = 2*p*pc*tc*(s3 - s1)
        yzc = 2*p*pc*t*(s3 - s1)
        
        
        
        xsigmax = x2c*xt*xt+y2c*yt*yt+z2c*zt*zt+xyc*xt*yt+xzc*xt*zt+yzc*yt*zt
        self.f0 = np.exp(h-0.5*xsigmax)
        # clip bk
        bk = np.clip(bk, -709.78, 709.78)
        self.f = np.exp(bk)+self.f0
        
        return self.f
    
    def frac_conv(self,x1,x2):return 2*np.abs(x1-x2)/(x1+x2)<self.eps_frac
    def dist_conv(self,x1,x2):return np.abs(x1-x2)<self.eps_dist
    def angle_conv(self,x1,x2):return np.abs(x1-x2)<self.eps_angle
    def calc_eps(self,parms):
        """
        calculate the loss function
        """
        #Decided not to include this extra step of convergence
        """
        if self.p_old is not None:
            if np.any((parms-self.parms_old)!=0):
                p_new = self.to_natural_paramaters(parms)
                h1,x1,y1,z1,bk1,wx1,wy1,wz1,t1,p1 = self.p_old[:10]
                h2,x2,y2,z2,bk2,wx2,wy2,wz2,t2,p2 = p_new[:10]
                #print self.p_old,p_new
                self.p_old = p_new
                
                converged = self.frac_conv(h1,h2) and self.frac_conv(bk1,bk2) 
                converged = converged and self.dist_conv(x1,x2) and self.dist_conv(y1,y2) and self.dist_conv(z1,z2)
                converged = converged and self.dist_conv(wx1,wx2) and self.dist_conv(wy1,wy2) and self.dist_conv(wz1,wz2)
                converged = converged and self.angle_conv(t1,t2) and self.angle_conv(p1,p2)
                if converged:
                    self.converged = True
                    return np.zeros(len(self.im),dtype=float)
        else:
            self.p_old = self.to_natural_paramaters(parms)
            self.parms_old = parms
        """
        return self.calc_f(parms)-self.im
    def calc_jac(self,parms):
        bk,h,xp,yp,zp,w1,w2,w3,pp,tp = parms
        t,p = self.to_sine(tp),self.to_sine(pp)
        ws1,ws2,ws3 = self.to_ws(w1),self.to_ws(w2),self.to_ws(w3)
        xc,yc,zc = self.to_center(xp,yp,zp)
        xt,yt,zt = self.x-xc,self.y-yc,self.z-zc
        p2 = p*p
        t2 = t*t
        tc2 = 1-t2
        pc2 = 1-p2
        tc= np.sqrt(tc2)
        pc= np.sqrt(pc2)
        s1,s2,s3 = 1./ws1,1./ws2,1./ws3
        x2c = pc2*tc2*s1 + t2*s2 + p2*tc2*s3
        y2c = pc2*t2*s1 + tc2*s2 + p2*t2*s3
        z2c = p2*s1 + pc2*s3
        xyc = 2*tc*t*(pc2*s1 - s2 + p2*s3)
        xzc = 2*p*pc*tc*(s3 - s1)
        yzc = 2*p*pc*t*(s3 - s1)
        xt2,xtyt,xtzt,yt2,ytzt,zt2 = xt*xt,xt*yt,xt*zt,yt*yt,yt*zt,zt*zt
        xsigmax = x2c*xt2+y2c*yt2+z2c*zt2+xyc*xtyt+xzc*xtzt+yzc*ytzt
        
        d,minw,maxw = self.delta_center,self.min_w,self.max_w
        
        
        f2 = np.exp(h-0.5*xsigmax)
        f1 =  np.exp(bk)+np.zeros(len(f2))
        e_xp,e_yp,e_zp = np.exp(-np.abs(xp)),np.exp(-np.abs(yp)),np.exp(-np.abs(zp))
        norm_xp = -d*e_xp/((1 + e_xp)*(1 + e_xp))
        norm_yp = -d*e_yp/((1 + e_yp)*(1 + e_yp))
        norm_zp = -d*e_zp/((1 + e_zp)*(1 + e_zp))
        f3 = (f2*(2*x2c*xt + xyc*yt + xzc*zt))*norm_xp
        f4 = (f2*(xt*xyc + 2*y2c*yt + yzc*zt))*norm_yp
        f5 = (f2*(xt*xzc + yt*yzc + 2*z2c*zt))*norm_zp
        f6 = (f2*(-pc2*tc2*xt2 - 2*pc2*t*tc*xtyt - pc2*t2*yt2 + 2*p*pc*tc*xtzt + 2*p*pc*t*ytzt - p2*zt2))*self.norm_w(w1,minw,maxw)
        f7 = (f2*(-t2*xt2 + 2*t*tc*xtyt - tc2*yt2))*self.norm_w(w2,minw,maxw)
        f8 = (f2*(-p2*tc2*xt2 - 2*p2*t*tc*xtyt - p2*t2*yt2 - 2*p*pc*tc*xtzt - 2*p*pc*t*ytzt - pc2*zt2))*self.norm_w(w3,minw,maxw)
        e_p = np.exp(-np.abs(pp)/2)
        norm_p = e_p/(1+e_p*e_p)
        f9 = f2*(s3-s1)*((2*pc2-1.)*(tc*xtzt + t*ytzt) + p*pc*(tc2*xt2 + 2*t*tc*xtyt + t2*yt2 - zt2))*norm_p
        e_t = np.exp(-np.abs(tp)/2)
        norm_t = e_t/(1+e_t*e_t)
        f10 = f2*((pc2 *s1 - s2 +  p2*s3)*(t *tc*(yt2 - xt2) - (t2 - tc2)*xtyt) + p* pc *(s1 - s3)* (t *xtzt - tc*ytzt))*norm_t
        
        self.jac = np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10],float).T
        
        return self.jac

    def norm_w(self,w,minw,maxw):
        if w>0:
            e_w = np.exp(-w)
            return 0.5*(maxw - minw)*e_w/(maxw*e_w+minw)**2
        else:
            e_w = np.exp(w)
            return 0.5*(maxw - minw)*e_w/(minw*e_w+maxw)**2

    def fit(self,eps_frac=10E-3,eps_dist=10E-3,eps_angle=10E-3):
        """
        This implements the Levenberg-Marquardt algorithm for 3D gaussian fitting.
        Stores the results in [height,x,y,z,background,width_1,width_2,width_3,sin_theta,sin_phi,error] = self.p
        """
        if len(self.p_)>len(self.im):
            self.success = False
        else:
            self.eps_frac,self.eps_dist,self.eps_angle = eps_frac,eps_dist,eps_angle
            parms0 = self.p_
            self.p_old = None
            parmsf,_ = leastsq(self.calc_eps,parms0,Dfun=self.calc_jac, maxfev=1000) # changed maxfev to block warning message
            #parmsf=parms0#####pisici
            self.p_ = parmsf
            self.to_natural_paramaters()
            self.center = self.p[1:4]
            self.success = True
    def get_im(self):
        self.calc_f(self.p_)
        return self.f0


def in_dim(x,y,z,xmax,ymax,zmax):
    keep = ((x>=0)&(x<xmax)&(y>=0)&(y<ymax)&(z>=0)&(z<zmax))>0
    return x[keep],y[keep],z[keep]

def closest_faster(xyz,ic,tree,rsearch = 6):
    dists_,nns_ = tree.query(xyz,distance_upper_bound=rsearch)
    return xyz[nns_==ic].T

class iter_fit_seed_points():
    def __init__(self,im,centers,radius_fit=5,min_delta_center=1.,max_delta_center=2.5,
                 n_max_iter = 5, max_dist_th=0.1,
                 min_w=0.5, max_w=4, init_w=1.5):
        """
        Given a set of seeds <centers> in a 3d image <im> iteratively 3d gaussian fit around the seeds (in order of brightness) 
        and subtract the gaussian signal.
        Retruns a numpy array of size Nx(height, x, y, z, width_x, width_y,width_z,background) where N~len(centers). Bad fits are disregarded.
        Warning: Generally a bit slow. In practice, the faster version fast_local_fit is used.
        """
        
        #internalize
        self.im = im
        self.radius_fit = radius_fit
        self.n_max_iter = n_max_iter
        self.max_dist_th = max_dist_th
        self.min_delta_center = min_delta_center
        self.max_delta_center = max_delta_center
        
        self.centers = centers.T
        self.z,self.x,self.y = centers
        self.zb,self.xb,self.yb = np.reshape(np.indices([self.radius_fit*2]*3)-self.radius_fit,[3,-1])
        keep = self.zb*self.zb+self.xb*self.xb+self.yb*self.yb<=self.radius_fit**2
        self.zb,self.xb,self.yb = self.zb[keep],self.xb[keep],self.yb[keep]
        self.zxyb = np.array([self.zb,self.xb,self.yb]).T
        self.sz,self.sx,self.sy = im.shape
        
        self.min_w = min_w
        self.max_w = max_w
        self.init_w = init_w
                    
    def firstfit(self):
        """
        Perform a first fit on the sample with the gaussian constrained close to the local maximum
        """
        
        if len(self.centers)>0:
            #fit the points in order of brightness and at each fit subtract the fitted signal
            self.ps = []
            self.ims_rec=[]
            self.im_subtr = np.array(self.im,dtype=np.float32)
            self.centers_fit = []
            self.success=[]
            self.centers_tree = KDTree(self.centers)
        
            centers_ = self.centers
            self.gparms = []
            for ic,(zc,xc,yc) in enumerate(centers_):
                z_keep,x_keep,y_keep = int(zc)+self.zb,int(xc)+self.xb,int(yc)+self.yb
                z_keep,x_keep,y_keep = in_dim(z_keep,x_keep,y_keep,self.sz,self.sx,self.sy)
                
                X_full = np.array([z_keep,x_keep,y_keep],dtype=int)
                center = [zc,xc,yc]
                z_keep,x_keep,y_keep = closest_faster(X_full.T,ic,self.centers_tree,rsearch = self.radius_fit*2)
                
                X = np.array([z_keep,x_keep,y_keep])
                im_ = self.im[z_keep,x_keep,y_keep]
                
                self.gparms
                
                obj = GaussianFit(im_,X,center=center,delta_center=self.min_delta_center,
                                  min_w=self.min_w, max_w=self.max_w, init_w=self.init_w)
                obj.fit()
                
                self.gparms.append([im_,X,center])
                n_p = len(obj.p)
                self.success.append(obj.success)
                if obj.success:
                    self.ps.append(obj.p)
                    self.centers_fit.append(obj.center)
                    z_keep,x_keep,y_keep = X_full
                    obj.x,obj.y,obj.z = X_full
                    im_rec = obj.get_im()
                    self.ims_rec.append(im_rec)
                    self.im_subtr[z_keep,x_keep,y_keep] -= im_rec
                else:
                    self.ims_rec.append(np.nan)
                    self.ps.append([np.nan]*n_p)
                    self.centers_fit.append([np.nan]*3)
                
        self.im_add = np.array(self.im_subtr)       
        
    def repeatfit(self):
             
        self.n_iter = 0
        num_seeds = len(self.centers)
        self.converged = np.zeros(len(self.centers),dtype=bool)
        self.dists = np.zeros(len(self.centers))+np.inf
        converged = np.all(self.converged)
        while not converged:
            
            self.success_old,self.centers_fit_old=np.array(self.success),np.array(self.centers_fit)
            #self.ps_old=np.array(self.ps)
            for ic,(zc,xc,yc) in enumerate(self.centers):
                if not self.converged[ic]:
                    #get modified image positions
                    z_keep,x_keep,y_keep = int(zc)+self.zb,int(xc)+self.xb,int(yc)+self.yb
                    z_keep,x_keep,y_keep = in_dim(z_keep,x_keep,y_keep,self.sz,self.sx,self.sy)
                    X = np.array([z_keep,x_keep,y_keep])
                    #get image, adding back the fit
                    im_ = self.im_add[z_keep,x_keep,y_keep]
                    
                    if self.success_old[ic]:
                        im_rec = self.ims_rec[ic]
                        im_=im_rec+im_ #add back the image
                        

                    delta_center = self.max_delta_center
                    obj = GaussianFit(im_,X,center=[zc,xc,yc],delta_center=delta_center,
                                        min_w=self.min_w, max_w=self.max_w, init_w=self.init_w)
                    obj.fit()
                    self.success[ic] = obj.success
                    if obj.success:
                        im_rec = obj.get_im()
                        self.ps[ic]=obj.p
                        self.centers_fit[ic]=obj.center
                        self.ims_rec[ic]=im_rec
                        self.im_add[z_keep,x_keep,y_keep] = im_-im_rec
            
            keep = (np.array(self.success)&np.array(self.success_old))>0
            self.dists[~keep]=0
            self.dists[keep]=np.sum((np.array(self.centers_fit_old)[keep]-np.array(self.centers_fit)[keep])**2,axis=-1)
            self.converged = self.dists<self.max_dist_th**2
            converged = np.all(self.converged)
            self.n_iter+=1
            converged = converged or (self.n_iter>self.n_max_iter)
        