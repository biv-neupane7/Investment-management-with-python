import pandas as pd
import numpy as np
import scipy.stats

def drawdown(return_series:pd.Series):
    """
    Takes a time-series of asset returns and 
    computes and returns a df that contains:
    the wealth index
    the previous peaks
    and percent drawdowns
    """

    wealth_index=1000*(1+return_series).cumprod() # Note that we do not know what is the return_series as of yet
    previous_peaks=wealth_index.cummax()
    drawdowns=(wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns})

def get_ffme_returns():
    
    """
    Load the FF dataset for the returns of the top and bottom deciles by MarketCap
    
    """
    me_m=pd.read_csv("data\\Portfolios_Formed_on_ME_monthly_EW.csv", index_col=0, header=0, na_values="-99.99")
    rets=(me_m[["Lo 10","Hi 10"]])/100
    rets.columns=["SmallCap","LargeCap"]
    rets.index=pd.to_datetime(rets.index, format="%Y%m").to_period("M")
    return rets
        
        
def get_hfi_returns():
    """
    Load and format the edhec hedge fund index returns
    
    """
    
    
    hfi=pd.read_csv("data\\edhec-hedgefundindices.csv", index_col=0, header=0, parse_dates=True)
    hfi=hfi/100
    hfi.index=hfi.index.to_period("M")
    return hfi

def get_ind_returns():
    """
    Load and format the ken french 30 industry portfolios value weighted monthly returns
    
    """
    ind=pd.read_csv("data\\ind30_m_vw_rets.csv", index_col=0, header=0, parse_dates=True)/100
    ind.index=pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns=ind.columns.str.strip()
    return ind

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    The periods per year is to be provided
    
    """
    compounded_growth=(1+r).prod() #Compounding
    n_periods=r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    
    """
    Annualizes a set of returns
    The periods per year is to be provided
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, rf_rate, periods_per_year):
    
    """
    Computes the annualized sharpe ratio of a set of returns"
   
    """
    # First, convert the annual risk free rate to per period
    rf_per_period= (1+rf_rate)**(1/periods_per_year)-1
    excess_ret=r-rf_per_period
    ann_ex_ret=annualize_rets(excess_ret, periods_per_year)
    ann_vol=annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol
    
    
    
def semideviation(r):
    
    """
    Returns semideviations aka negative semi deviation of r 
    r must be a Series or a DataFrame
    
    """
    
    is_negative=r<0
    return r[is_negative].std(ddof=0)

def skewness(r):
    
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series 
    
    """
    
    demeaned_r=r-r.mean()
    # use the pop. st. deviation, so set the dof=0
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series 
    
    """
    
    demeaned_r=r-r.mean()
    # use the pop. st. deviation, so set the dof=0
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**4).mean()
    return exp/sigma_r**4



def is_normal(r, level =0.01): 
    
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypo of normality is accepted, False otherwise
    
    The level inside the paranthesis is basically saying that we wanna have at least 1% of level of confidence that this thing is normal
    This is nothing more than saying that the p-value should be more than 1% to not reject the null hypo 
    """
    
    statistic, p_value= scipy.stats.jarque_bera(r) #Unpacking because the method on the right results in tuple on the left
    return p_value > level 

    
    
import numpy as np    
def var_historic(r, level=5):
    """
    Returns the Historic VaR at a specified level i.e
    returns the number such that "level" percent of returns fall below that number,
    and the (100-level) percent are above. 
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or a DataFrame")

        
from scipy.stats import norm
def var_gaussian(r, level=5):
    """
    Returns the parametric Gaussian VaR of Series or a DataFrame

    """
    #first, compute z score assuming it was gaussian
    z_score=norm.ppf(level/100)
    return -(r.mean()+z_score*r.std(ddof=0))


def var_corfish(r, level=5, modified=True):
    """
    Returns the parametric Gaussian VaR of Series or a DataFrame
    If "modified" is True, then the modified VaR is returned, using the 
    Cornish-Fisher modification

    """
    
    #first, compute z score assuming it was gaussian
    z_score=norm.ppf(level/100)
    if modified:
        # modify the z score based on observed skewness and kurtosis
        s=skewness(r)
        k=kurtosis(r)
        z_score=(z_score+
           (z_score**2-1)*s/6 +
           (z_score**3- 3*z_score)*(k-3)/24 -
           (2*z_score**3- 5*z_score)*(s**2)/36
          )
        
    return -(r.mean()+z_score*r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    Returns the CVaR of a Series or a DataFrame 
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    elif isinstance(r, pd.Series):
        is_beyond= r<=-var_historic(r, level=level)
        return -r[is_beyond].mean()
    else:
        raise TypeError("Expected r to be a Series or a DataFrame")
        
  

def portfolio_return(weights, returns):
    """
    Weights->> to ->> returns
    """
    return weights.T @ returns   # weight vector transposed and multiplied by the returns vector


def portfolio_vol(weights, covmat):
    
    """
    weights ->> vol
    """
    return (weights.T @ covmat @ weights)**0.5



def plot_ef2(n_points, er, cov):
    
    """
    Plots the 2 asset eff frontier
    """
    
    if er.shape[0] !=2 or er.shape[0] !=2:
        raise ValueError("plot_ef2 can only plot 2 asset frontiers")
    weights=[np.array([w, 1-w]) for w in np.linspace(0,1, n_points)] # linspace is just lineraly(equally) spaced points bet 2 numbers
    rets=[portfolio_return(w,er) for w in weights]
    vols=[portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({"Return":rets, "Volatility":vols})
    
    return ef.plot.line(x="Volatility",y="Return", style=".-")


from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
    
    """
    target_return -> W
    """
    n=er.shape[0]
    init_guess= np.repeat(1/n, n)
    bounds=((0.0, 1.0),)*n
    return_is_target={
        "type":"eq",
        "args":(er,),
        "fun": lambda weights, er: target_return-portfolio_return(weights, er) }
    weights_sum_to_1={
        "type":"eq",
        "fun": lambda weights: np.sum(weights)-1}
    results= minimize(portfolio_vol,init_guess, args=(cov,), method="SLSQP",
                options={"disp":False},
                constraints=(return_is_target, weights_sum_to_1),
                bounds=bounds)
    return results.x



def optimal_weights(n_points, er, cov):
    
    """
    Generates a list of weights to run the optimizer on to minimize the vol
    """
    
    target_rs=np.linspace(er.min(), er.max(), n_points) # First generating list of target returns to feed the optimizer that will
                                                        # eventually use this info to compute optimal weights
    
    # We previously had written a code to generate optimal weights. However, that was for 1 set of weights but now, we need 
    # a series of weights from the series of target returns given by the above variable
    
    weights=[minimize_vol(target_return, er, cov) for target_return in target_rs] # This code is basically saying " Loop through
                                                            # every one of "target_rs" series, run the optimizer on every one of 
                                                    # "target_return" and give me back the set of optimal weights
    return weights
    

    
def msr(rf_rate, er, cov):  # Change-> removed target return and replace it with rf_rate. It is basically saying that the msr
                            # portfolio is a function of rf_rate
        """
        Returns the weights of the portfolio that gives you the max sharpe ratio given the rf rate, ER and the cov matrix
        """
        n=er.shape[0]
        init_guess=np.repeat(1/n, n)
        bounds=((0.0,1.0),)*n
        
        # We are not targetting any returns right now so that is why there shall be no "return_is_target" constraint
        
        # However, we still want the weights to sum to 1
        
        weights_sum_to_1 = {
            "type":"eq",
            "fun": lambda weights: np.sum(weights)-1
        }
        
        # Note that we are not trying to minimize the portfolio volatility. We are rather trying to maximize the Sharpe ratio.
        # This is why in the code below, the objective function is gonna change. Instead of "minimize(ak.portfolio_vol)", it 
        # shall be "minimize(neg_sharpe_ratio)". 
        
        # The logic here is that if we minimize the negative sharpe ratio, we shall automatically be maximizing the sharpe 
        # ratio
        # So, therefore now, lets define the neg_sharpe_ratio function
        
        def neg_sharpe_ratio(weights, rf_rate, er, cov):
            """
            Returns the negative of the Sharpe ratio, given weights
            """
            
            rets= portfolio_return(weights, er)
            vol= portfolio_vol(weights, cov)
            
            return -(rets-rf_rate)/vol
        
        results= minimize(neg_sharpe_ratio, init_guess, args=(rf_rate,er,cov,), method="SLSQP",
                         options={"disp":False}, constraints=(weights_sum_to_1), bounds=bounds
                         )
        return results.x
    

    
def gmv(cov):
    
    """
    Returns the weight of the GMV portfolio given the covariance matrix
    """
    n=cov.shape[0]
    return msr(0, np.repeat(1,n),cov)
    
def plot_ef(n_points, er, cov, show_cml=False,show_ew=False, show_gmv=False, style=".-", rf_rate=0):
    
    """
    Plots the n asset eff frontier with tangency point and line
    """
    
    weights=optimal_weights(n_points, er, cov) # linspace is just lineraly(equally) spaced points between 2 numbers
    rets=[portfolio_return(w, er) for w in weights]
    vols=[portfolio_vol(w, cov) for w in weights]
    ef=pd.DataFrame({"Returns":rets, "Volatility":vols})
    
    ax=ef.plot.line(x="Volatility",y="Returns", style=style)
    
    if show_ew:
        n=er.shape[0]
        w_ew=np.repeat(1/n,n)
        r_ew=portfolio_return(w_ew, er)
        vol_ew=portfolio_vol(w_ew, cov)
        
        # Display the equally weighted portfolio
        
        ax.plot([vol_ew],[r_ew], color="goldenrod", marker="o", markersize=12)
    
    if show_gmv:
        w_gmv=gmv(cov)
        r_gmv=portfolio_return(w_gmv, er)
        vol_gmv=portfolio_vol(w_gmv, cov)
        
        #Display the gmv portfolio
        
        ax.plot([vol_gmv],[r_gmv], color="midnightblue", marker="o",markersize=12)
    
    if show_cml:
        ax.set_xlim(left=0)
        w_msr=msr(rf_rate, er, cov)
        r_msr=portfolio_return(w_msr, er)
        vol_msr=portfolio_vol(w_msr, cov)
        
        # Add CML
        
        cml_x=[0, vol_msr]
        cml_y=[rf_rate, r_msr]
        ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", markersize=12, linewidth=2)
        
    return ax

