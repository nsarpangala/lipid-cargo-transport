import numpy as np
"""This script has Klump expressions.

    Paper: https://doi.org/10.1073/pnas.0507363102
"""

'''after the assumption of linear decrease in on rate and increase in off rate as fn of n'''
def bound_N(pi_ad,epsilon,N):
    """ Returns average number of bound motors

        Parameters:
            pi_ad effective binding rate
            epsilon off rate
            N number of motors

        Returns
            float: average number of bound motors
        """
    return (pi_ad/epsilon)*(1+(pi_ad/epsilon))**(N-1)*N/((1+(pi_ad/epsilon))**N-1)

def klump_run(N,v,pi_ad,eps):
    """ Returns run-length of cargo motor system.

        Parameters:
            N (int) total number of motors
            v (float) velocity of motors
            pi_ad (float) effective binding rate
            eps (float) off rate of motors

        Returns:
            float: Run length of cargo
        """
    return (v/(N*pi_ad))*((1+pi_ad/eps)**N-1)

assert abs(bound_N(0.5,2.0,16)-3.2926)<1e-2
assert abs(klump_run(16,0.8e-6,0.5,2.0)-3.452e-6)<1e-8


''' taking measured values of on rate and off rate'''
def more_accurate_klump(v,eps,pi_array,N_eff):
    N_eff_int=int(round(N_eff))
    sum_rat=1
    if N_eff_int<1:
        print('error')
    else:
        for n_boun in range(1,N_eff_int):
            prod=1
            for ii in range(1,n_boun+1):
                if ii>len(pi_array):
                    pi_o_n=pi_array[len(pi_array)-1]
                else:
                    pi_o_n=pi_array[ii-1]
                    
                prod=prod*(N_eff_int-ii)*pi_o_n/((ii+1)*eps)
            sum_rat+=prod 
            
    run_length=sum_rat*v/eps
    return run_length










''' Effective on rate and number of motors'''


def access_probability(H,L_mot,R):
    #alpha_max=0.5*(1-(H-L_mot)/R)
    #alpha_min=0.5*(1-(H**2+R**2-L_mot**2)/(2*H*R))
    #return 0.5*(alpha_max+alpha_min)
    
    return 1.884955592153876e-14/(4*np.pi*R**2)


def eff_values(R,d_arr,N,kon,access_a,t_off):
    del_theta=np.sqrt(2*d_arr*t_off)/R


#     theta_max=np.arccos((H-L_mot)/R)
#     theta_min=np.arccos((H**2+R**2-L_mot**2)/(2*H*R))

#     a_max=2*np.pi*R**2*(1-np.cos(theta_max))
#     a_min=2*np.pi*R**2*(1-np.cos(theta_min))

#     a_av=0.5*(a_max+a_min)
    
#     access_a=1.884955592153876e-14
#     a_av=access_a
    
    av_theta=np.arccos(1-access_a/(2*np.pi*R**2))


    inf_theta=av_theta+del_theta

    inf_theta[inf_theta>np.pi]=np.pi
    inf_area=2*np.pi*R**2*(1-np.cos(inf_theta))

    eff_on_rates=kon*access_a/inf_area
    eff_N=(N-1)*inf_area/(4*np.pi*R**2)+1

    return [eff_N,eff_on_rates]



