from enum import Enum
import stat
from networkx import display
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, final
from scipy.stats import norm
from fixedincomelib.analytics.european_options import (
    CallOrPut,
    SimpleMetrics,
    EuropeanOptionAnalytics,
)


class SabrMetrics(Enum):

    # parameters
    ALPHA = "alpha"
    BETA = "beta"
    NU = "nu"
    RHO = "rho"

    # risk
    DALPHA = "dalpha"
    DLNSIGMA = "dlnsigma"
    DNORMALSIGMA = "dnormalsigma"
    DBETA = "dbeta"
    DRHO = "drho"
    DNU = "dnu"
    DFORWARD = "dforward"
    DSTRIKE = "dstrike"
    DTTE = "dtte"
    DSTRIKESTRIKE = "dstrikestrike"

    # (alpha, beta, nu, rho, forward, strike, tte) => \sigma_k
    D_LN_SIGMA_D_FORWARD = "d_ln_sigma_d_forward"
    D_LN_SIGMA_D_STRIKE = "d_ln_sigma_d_strike"
    D_LN_SIGMA_D_TTE = "d_ln_sigma_d_tte"
    D_LN_SIGMA_D_ALPHA = "d_ln_sigma_d_alpha"
    D_LN_SIGMA_D_BETA = "d_ln_sigma_d_beta"
    D_LN_SIGMA_D_NU = "d_ln_sigma_d_nu"
    D_LN_SIGMA_D_RHO = "d_ln_sigma_d_rho"
    D_LN_SIGMA_D_STRIKESTRIKE = "d_ln_sigma_d_strike_strike"

    # (\sigma_ln_atm, f, tte, beta, nu, rho) => alpha
    D_ALPHA_D_LN_SIGMA_ATM = "d_alpha_d_ln_sigma_atm"
    D_ALPHA_D_FORWARD = "d_alpha_d_forward"
    D_ALPHA_D_TTE = "d_alpha_d_tte"
    D_ALPHA_D_BETA = "d_alpha_d_beta"
    D_ALPHA_D_NU = "d_alpha_d_nu"
    D_ALPHA_D_RHO = "d_alpha_d_rho"

    # (alpha, beta, nu, rho, f, tte) => \sigma_n_atm
    D_NORMAL_SIGMA_D_ALPHA = "d_normal_sigma_d_alpha"
    D_NORMAL_SIGMA_D_BETA = "d_normal_sigma_d_beta"
    D_NORMAL_SIGMA_D_NU = "d_normal_sigma_d_nu"
    D_NORMAL_SIGMA_D_RHO = "d_normal_sigma_d_rho"
    D_NORMAL_SIGMA_D_FORWARD = "d_normal_sigma_d_forward"
    D_NORMAL_SIGMA_D_TTE = "d_normal_sigma_d_tte"
    D_ALPHA_D_NORMAL_SIGMA_ATM = "d_alpha_d_normal_sigma_atm"

    @classmethod
    def from_string(cls, value: str) -> "SabrMetrics":
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError as e:
            raise ValueError(f"Invalid token: {value}") from e

    def to_string(self) -> str:
        return self.value


class SABRAnalytics:

    EPSILON = 1e-6

    ### parameters conversion

    # solver to back out lognormal vol from alpha and sensitivities
    # please implement the _vol_and_risk function to make this work
    @staticmethod
    def lognormal_vol_from_alpha(
        forward: float,
        strike: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
    ) -> Dict[SabrMetrics | SimpleMetrics, float]:

        res: Dict[Any, float] = {}

        ln_sigma, risks = SABRAnalytics._vol_and_risk(
            forward + shift, strike + shift, time_to_expiry, alpha, beta, rho, nu, calc_risk
        )
        res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL] = ln_sigma

        if len(risks) == 0:
            return res

        res.update(risks)
        return res

    @staticmethod
    def alpha_from_atm_lognormal_sigma(
        forward: float,
        time_to_expiry: float,
        sigma_atm_lognormal: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
        max_iter: Optional[int] = 50,
        tol: Optional[float] = 1e-12,
    ) -> Dict[SabrMetrics, float]:

        if forward + shift <= 0.0:
            raise ValueError("forward must be > 0")
        if time_to_expiry < 0.0:
            raise ValueError("time_to_expiry must be >= 0")
        if sigma_atm_lognormal <= 0.0:
            raise ValueError("sigma_atm_lognormal must be > 0")
        if abs(rho) >= 1.0:
            raise ValueError("rho must be in (-1,1)")
        if nu < 0.0:
            raise ValueError("nu must be >= 0")
        if not (0.0 <= beta <= 1.0):
            raise ValueError("beta should be in [0,1] for standard SABR usage")

        # newton + bisec fallback
        # root finding
        # f = F(alpha, theta) - ln_sigma = 0
        # where F is lognormal_vol_from_alpha
        # alpha^* = alpha(ln_sigma, theta)

        this_res = None
        alpha = sigma_atm_lognormal * (forward + shift) ** (1.0 - beta)
        lower_bound = 0.0
        upper_bound = alpha * 10.0
        for _ in range(max_iter):
            # implement your newton step here to update alpha
            this_res = SABRAnalytics.lognormal_vol_from_alpha(forward,forward,time_to_expiry,alpha, beta, rho, nu, shift, True)
            implied_atm_lognormal_sigma = this_res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            error = implied_atm_lognormal_sigma - sigma_atm_lognormal
            slope = this_res[SabrMetrics.D_LN_SIGMA_D_ALPHA]

            if error > 0: upper_bound = min(alpha, upper_bound)
            else: lower_bound = max(alpha, lower_bound)
            if abs(error) < tol: break

            alpha_est = alpha - error / slope
            # fallback to bisection method if newton step goes out of bounds
            if alpha_est < lower_bound or alpha_est > upper_bound:
                alpha = (lower_bound + upper_bound) * .5
            else:
                alpha = alpha_est

        else:
            raise RuntimeError("alpha_from_atm_lognormal_sigma: Newton did not converge")

        res: Dict[SabrMetrics, float] = {SabrMetrics.ALPHA: alpha}

        if calc_risk:

            # dalphad...
            # alpha^* = alpha(ln_sigma, theta, target_ln_sigma)
            # F(alpha(ln_sigma, theta), theta) = target_ln_sigma
            # using implicit function theorem
            # df/dalpha * dalpha/dln_sigma = 1 =>             dalpha / dln_sigma = 1 / df/dalpha
            res[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM] = 1.0 / this_res[SabrMetrics.D_LN_SIGMA_D_ALPHA]
            # df/dalpha * dalpha/dtheta  + df/dtheta = 0 =>  dalpha / dtheta = - df/dtheta / df/dalpha
            for key, risk in [
                (SabrMetrics.D_ALPHA_D_TTE, SabrMetrics.D_LN_SIGMA_D_TTE),
                (SabrMetrics.D_ALPHA_D_BETA, SabrMetrics.D_LN_SIGMA_D_BETA),
                (SabrMetrics.D_ALPHA_D_NU, SabrMetrics.D_LN_SIGMA_D_NU),
                (SabrMetrics.D_ALPHA_D_RHO, SabrMetrics.D_LN_SIGMA_D_RHO),
            ]:
                res[key] = -this_res[risk] / this_res[SabrMetrics.D_LN_SIGMA_D_ALPHA]
            res[SabrMetrics.D_ALPHA_D_FORWARD] = - (this_res[SabrMetrics.D_LN_SIGMA_D_FORWARD] + this_res[SabrMetrics.D_LN_SIGMA_D_STRIKE]) / this_res[SabrMetrics.D_LN_SIGMA_D_ALPHA]
        
        return res

    # conversion to alpha from normal atm vol
    @staticmethod
    def alpha_from_atm_normal_sigma(
        forward: float,
        time_to_expiry: float,
        sigma_atm_normal: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: bool = False,
        max_iter: int = 50,
        tol: float = 1e-8,
    ) -> Dict[SabrMetrics, float]:

        # at atm, from nv vol to ln vol
        # please check the functions in 'EuropeanOptionAnalytics.py'

        # compute implied log normal vol
        res_log_normal_atm_vol = EuropeanOptionAnalytics.normal_vol_to_lognormal_vol(forward, forward, time_to_expiry, sigma_atm_normal, calc_risk, shift, tol)
        log_normal_atm_vol = res_log_normal_atm_vol[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
        res_alpha = SABRAnalytics.alpha_from_atm_lognormal_sigma(forward, time_to_expiry, log_normal_atm_vol, beta, rho, nu, shift, calc_risk, max_iter, tol)

        # risk aggregation
        final_res = {SabrMetrics.ALPHA: res_alpha[SabrMetrics.ALPHA]}

        if calc_risk:
            # d_alpha_d...
            # ln_sigma = V(n_sigma, F, K, TTE)
            # F(alpha(V(n_sigma, F, K, TTE), b, r, n, F, TTE), b, r, n, F, K=F, TTE) = V(n_sigma, F, K, TTE)
            # using implicit function theorem
            # df/dalpha * dalpha/dn_sigma = dln_sigma/dn_sigma => dalpha/dn_sigma = dln_sigma/dn_sigma * 1 / df/dalpha = dln_sigma/dn_sigma * dalpha/dln_sigma
            final_res[SabrMetrics.D_ALPHA_D_NORMAL_SIGMA_ATM] = res_log_normal_atm_vol[SimpleMetrics.D_LN_VOL_D_N_VOL] * res_alpha[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM]
            # for beta, rho, nu: dln_sigma/dalpha * dalpha/dtheta + dln_sigma/dtheta = 0 => dalpha/dtheta = - dln_sigma/dtheta / dln_sigma/dalpha
            for key in [
                SabrMetrics.D_ALPHA_D_BETA,
                SabrMetrics.D_ALPHA_D_NU,
                SabrMetrics.D_ALPHA_D_RHO
            ]:
                final_res[key] = res_alpha[key]
            
            final_res[SabrMetrics.D_ALPHA_D_TTE] = res_alpha[SabrMetrics.D_ALPHA_D_TTE] + res_log_normal_atm_vol[SimpleMetrics.D_LN_VOL_D_TTE] * res_alpha[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM]
            final_res[SabrMetrics.D_ALPHA_D_FORWARD] = res_alpha[SabrMetrics.D_ALPHA_D_FORWARD] + res_alpha[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM] *(res_log_normal_atm_vol[SimpleMetrics.D_LN_VOL_D_FORWARD] + res_log_normal_atm_vol[SimpleMetrics.D_LN_VOL_D_STRIKE])
        return final_res

    ### option pricing

    @staticmethod
    def european_option_alpha(
        forward: float,
        strike: float,
        time_to_expiry: float,
        opt_type: CallOrPut,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
    ):

        ### pv
        ln_sigma_and_sensitivities = SABRAnalytics.lognormal_vol_from_alpha(
            forward, strike, time_to_expiry, alpha, beta, rho, nu, shift, calc_risk
        )
        ln_iv = ln_sigma_and_sensitivities[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
        value_and_sensitivities = EuropeanOptionAnalytics.european_option_log_normal(
            forward + shift, strike + shift, time_to_expiry, ln_iv, opt_type, calc_risk
        )

        ### risk(analytic)
        if calc_risk:
            ## first order risks
            dvdsigma = value_and_sensitivities[SimpleMetrics.VEGA]
            value_and_sensitivities.pop(SimpleMetrics.VEGA)
            # delta
            value_and_sensitivities[SimpleMetrics.DELTA] += (
                dvdsigma * ln_sigma_and_sensitivities[SabrMetrics.D_LN_SIGMA_D_FORWARD]
            )
            # theta
            value_and_sensitivities[SimpleMetrics.THETA] -= (
                dvdsigma * ln_sigma_and_sensitivities[SabrMetrics.D_LN_SIGMA_D_TTE]
            )
            # sabr alpha/beta/nu/rho
            for key, risk in [
                (SabrMetrics.DALPHA, SabrMetrics.D_LN_SIGMA_D_ALPHA),
                (SabrMetrics.DBETA, SabrMetrics.D_LN_SIGMA_D_BETA),
                (SabrMetrics.DRHO, SabrMetrics.D_LN_SIGMA_D_RHO),
                (SabrMetrics.DNU, SabrMetrics.D_LN_SIGMA_D_NU),
            ]:
                value_and_sensitivities[key] = dvdsigma * ln_sigma_and_sensitivities[risk]
            # strike
            value_and_sensitivities[SimpleMetrics.STRIKE_RISK] += (
                dvdsigma * ln_sigma_and_sensitivities[SabrMetrics.D_LN_SIGMA_D_STRIKE]
            )

            ## second order risk (bump reval)
            v_base = value_and_sensitivities[SimpleMetrics.PV]
            # strike
            res_up = SABRAnalytics.lognormal_vol_from_alpha(
                forward, strike + SABRAnalytics.EPSILON, time_to_expiry, alpha, beta, rho, nu, shift
            )
            vol_up = res_up[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            v_up = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift,
                strike + shift + SABRAnalytics.EPSILON,
                time_to_expiry,
                vol_up,
                opt_type,
            )[SimpleMetrics.PV]

            res_dn = SABRAnalytics.lognormal_vol_from_alpha(
                forward, strike - SABRAnalytics.EPSILON, time_to_expiry, alpha, beta, rho, nu, shift
            )
            vol_dn = res_dn[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            v_dn = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift,
                strike + shift - SABRAnalytics.EPSILON,
                time_to_expiry,
                vol_dn,
                opt_type,
            )[SimpleMetrics.PV]
            value_and_sensitivities[SimpleMetrics.STRIKE_RISK_2] = (v_up - 2 * v_base + v_dn) / (
                SABRAnalytics.EPSILON**2
            )

            # gamma
            res_up = SABRAnalytics.lognormal_vol_from_alpha(
                forward + SABRAnalytics.EPSILON, strike, time_to_expiry, alpha, beta, rho, nu, shift
            )
            vol_up = res_up[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            v_up = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift + SABRAnalytics.EPSILON,
                strike + shift,
                time_to_expiry,
                vol_up,
                opt_type,
            )[SimpleMetrics.PV]
            res_dn = SABRAnalytics.lognormal_vol_from_alpha(
                forward - SABRAnalytics.EPSILON, strike, time_to_expiry, alpha, beta, rho, nu, shift
            )
            vol_dn = res_dn[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
            v_dn = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift - SABRAnalytics.EPSILON,
                strike + shift,
                time_to_expiry,
                vol_dn,
                opt_type,
            )[SimpleMetrics.PV]
            value_and_sensitivities[SimpleMetrics.GAMMA] = (v_up - 2 * v_base + v_dn) / (
                SABRAnalytics.EPSILON**2
            )

        return value_and_sensitivities

    # Given function
    @staticmethod
    def european_option_ln_sigma(
        forward: float,
        strike: float,
        time_to_expiry: float,
        opt_type: CallOrPut,
        ln_sigma_atm: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
    ):

        ### pv
        alpha_and_sensitivities = SABRAnalytics.alpha_from_atm_lognormal_sigma(
            forward, time_to_expiry, ln_sigma_atm, beta, rho, nu, shift, calc_risk
        )
        alpha = alpha_and_sensitivities[SabrMetrics.ALPHA]
        value_and_sensitivities = SABRAnalytics.european_option_alpha(
            forward, strike, time_to_expiry, opt_type, alpha, beta, rho, nu, shift, calc_risk
        )

        ### risk
        if calc_risk:
            ## first order risks
            dvdalpha = value_and_sensitivities[SabrMetrics.DALPHA]
            value_and_sensitivities.pop(SabrMetrics.DALPHA)

            # delta
            value_and_sensitivities[SimpleMetrics.DELTA] += (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_FORWARD]
            )
            # theta
            value_and_sensitivities[SimpleMetrics.THETA] -= (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_TTE]
            )
            # ln_sigma
            value_and_sensitivities[SabrMetrics.DLNSIGMA] = (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_LN_SIGMA_ATM]
            )
            # sabr beta/rho/nu
            for key, risk in [
                (SabrMetrics.DBETA, SabrMetrics.D_ALPHA_D_BETA),
                (SabrMetrics.DRHO, SabrMetrics.D_ALPHA_D_RHO),
                (SabrMetrics.DNU, SabrMetrics.D_ALPHA_D_NU),
            ]:
                value_and_sensitivities[key] += dvdalpha * alpha_and_sensitivities[risk]

            ## second order risk (bump reval)
            v_base = value_and_sensitivities[SimpleMetrics.PV]

            # gamma
            res_up = SABRAnalytics.alpha_from_atm_lognormal_sigma(
                forward + SABRAnalytics.EPSILON, time_to_expiry, ln_sigma_atm, beta, rho, nu, shift
            )
            alpha_up = res_up[SabrMetrics.ALPHA]
            v_up = SABRAnalytics.european_option_alpha(
                forward + SABRAnalytics.EPSILON,
                strike,
                time_to_expiry,
                opt_type,
                alpha_up,
                beta,
                rho,
                nu,
                shift,
            )[SimpleMetrics.PV]
            res_dn = SABRAnalytics.alpha_from_atm_lognormal_sigma(
                forward - SABRAnalytics.EPSILON, time_to_expiry, ln_sigma_atm, beta, rho, nu, shift
            )
            alpha_dn = res_dn[SabrMetrics.ALPHA]
            v_dn = SABRAnalytics.european_option_alpha(
                forward - SABRAnalytics.EPSILON,
                strike,
                time_to_expiry,
                opt_type,
                alpha_dn,
                beta,
                rho,
                nu,
                shift,
            )[SimpleMetrics.PV]
            value_and_sensitivities[SimpleMetrics.GAMMA] = (v_up - 2 * v_base + v_dn) / (
                SABRAnalytics.EPSILON**2
            )

        return value_and_sensitivities

    # European call/put SABR risk with normal vol input, please implement this function with european_option_alpha api
    @staticmethod
    def european_option_normal_sigma(
        forward: float,
        strike: float,
        time_to_expiry: float,
        opt_type: CallOrPut,
        normal_sigma_atm: float,
        beta: float,
        rho: float,
        nu: float,
        shift: Optional[float] = 0.0,
        calc_risk: Optional[bool] = False,
    ):
        """
        Please implement this function with european_option_alpha api

        """
        value_and_sensitivities = {}
        ### pv
        alpha_and_sensitivities = SABRAnalytics.alpha_from_atm_normal_sigma(
            forward, time_to_expiry, normal_sigma_atm, beta, rho, nu, shift, calc_risk
        )
        alpha = alpha_and_sensitivities[SabrMetrics.ALPHA]
        value_and_sensitivities = SABRAnalytics.european_option_alpha(
            forward, strike, time_to_expiry, opt_type, alpha, beta, rho, nu, shift, calc_risk
        )

        ### risk
        if calc_risk:
            ## first order risks
            dvdalpha = value_and_sensitivities[SabrMetrics.DALPHA]
            value_and_sensitivities.pop(SabrMetrics.DALPHA)

            # delta
            value_and_sensitivities[SimpleMetrics.DELTA] += (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_FORWARD]
            )
            # theta
            value_and_sensitivities[SimpleMetrics.THETA] -= (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_TTE]
            )
            # normal_sigma
            value_and_sensitivities[SabrMetrics.DNORMALSIGMA] = (
                dvdalpha * alpha_and_sensitivities[SabrMetrics.D_ALPHA_D_NORMAL_SIGMA_ATM]
            )

            # sabr beta/rho/nu
            for key, risk in [
                (SabrMetrics.DBETA, SabrMetrics.D_ALPHA_D_BETA),
                (SabrMetrics.DRHO, SabrMetrics.D_ALPHA_D_RHO),
                (SabrMetrics.DNU, SabrMetrics.D_ALPHA_D_NU),
            ]:
                value_and_sensitivities[key] += dvdalpha * alpha_and_sensitivities[risk]

            # second order risk (bump reval)
            v_base = value_and_sensitivities[SimpleMetrics.PV]

            # gamma
            res_up = SABRAnalytics.alpha_from_atm_normal_sigma(
                forward + SABRAnalytics.EPSILON, time_to_expiry, normal_sigma_atm, beta, rho, nu, shift
            )
            alpha_up = res_up[SabrMetrics.ALPHA]
            v_up = SABRAnalytics.european_option_alpha(
                forward + SABRAnalytics.EPSILON,
                strike,
                time_to_expiry,
                opt_type,
                alpha_up,
                beta,
                rho,
                nu,
                shift,
            )[SimpleMetrics.PV]
            res_dn = SABRAnalytics.alpha_from_atm_normal_sigma(
                forward - SABRAnalytics.EPSILON, time_to_expiry, normal_sigma_atm, beta, rho, nu, shift
            )
            alpha_dn = res_dn[SabrMetrics.ALPHA]
            v_dn = SABRAnalytics.european_option_alpha(
                forward - SABRAnalytics.EPSILON,
                strike,
                time_to_expiry,
                opt_type,
                alpha_dn,
                beta,
                rho,
                nu,
                shift,
            )[SimpleMetrics.PV]
            value_and_sensitivities[SimpleMetrics.GAMMA] = (v_up - 2 * v_base + v_dn) / (
                SABRAnalytics.EPSILON**2
            )

        return value_and_sensitivities

   

    ### helpers

    @staticmethod
    def w2_risk(F, K, T, a, b, r, n) -> Dict:

        risk = {}

        risk[SabrMetrics.DALPHA] = (1 - b) ** 2 / 12 * a / (F * K) ** (1 - b) + b * r * n / (
            4 * (F * K) ** ((1 - b) / 2)
        )
        risk[SabrMetrics.DBETA] = (
            1 / 12 * (b - 1) * a**2 * (F * K) ** (b - 1)
            + 1 / 24 * (b - 1) ** 2 * a**2 * (F * K) ** (b - 1) * np.log(F * K)
            + 1 / 4 * a * r * n * (F * K) ** ((b - 1) / 2)
            + 1 / 8 * a * b * r * n * (F * K) ** ((b - 1) / 2) * np.log(F * K)
        )
        risk[SabrMetrics.DRHO] = 1 / 4 * a * b * n * (F * K) ** ((b - 1) / 2) - 1 / 4 * n**2 * r
        risk[SabrMetrics.DNU] = (
            1 / 4 * a * b * r * (F * K) ** ((b - 1) / 2) + 1 / 6 * n - 1 / 4 * r**2 * n
        )
        risk[SabrMetrics.DFORWARD] = (b - 1) ** 3 / 24 * a**2 * (F * K) ** (
            b - 2
        ) * K + a * r * n * b * (b - 1) / 8 * K ** ((b - 1) / 2) * F ** ((b - 3) / 2)

        risk[SabrMetrics.DSTRIKE] = (b - 1) ** 3 / 24 * a**2 * F ** (b - 1) * K ** (
            b - 2
        ) + a * b * r * n * (b - 1) / 8 * F ** ((b - 1) / 2) * K ** ((b - 3) / 2)

        risk[SabrMetrics.DSTRIKESTRIKE] = (b - 1) ** 3 / 24 * a**2 * (b - 2) * F ** (
            b - 1
        ) * K ** (b - 3) + a * b * r * n / 16 * (b - 1) * (b - 3) * F ** ((b - 1) / 2) * K ** (
            (b - 5) / 2
        )

        return risk

    @staticmethod
    def w1_risk(F, K, T, a, b, r, n) -> Dict:

        log_FK = np.log(F / K)

        risk = {}
        risk[SabrMetrics.DALPHA] = 0.0
        risk[SabrMetrics.DBETA] = (b - 1) / 12.0 * log_FK**2 + (b - 1) ** 3 / 480 * log_FK**4
        risk[SabrMetrics.DRHO] = 0.0
        risk[SabrMetrics.DNU] = 0.0
        risk[SabrMetrics.DFORWARD] = (b - 1) ** 2 / 12 * log_FK / F + (
            b - 1
        ) ** 4 / 480 / F * log_FK**3
        risk[SabrMetrics.DSTRIKE] = (
            -((b - 1) ** 2) / 12 * log_FK / K - (b - 1) ** 4 / 480 / K * log_FK**3
        )
        risk[SabrMetrics.DSTRIKESTRIKE] = (
            (b - 1) ** 2 / 12 / K**2
            + (b - 1) ** 2 / 12 * log_FK / K**2
            + (b - 1) ** 4 / 160 * log_FK**2 / K**2
            + (b - 1) ** 4 / 480 * log_FK**3 / K**2
        )

        return risk

    @staticmethod
    def z_risk(F, K, T, a, b, r, n) -> Dict:

        log_FK = np.log(F / K)
        fk = (F * K) ** ((1 - b) / 2)
        # z = n / a * log_FK * fk

        risk = {}
        risk[SabrMetrics.DALPHA] = -n / a * log_FK * fk / a
        risk[SabrMetrics.DBETA] = -1.0 / 2 * n / a * log_FK * fk * np.log(F * K)
        risk[SabrMetrics.DRHO] = 0.0
        risk[SabrMetrics.DNU] = 1.0 / a * log_FK * fk
        risk[SabrMetrics.DFORWARD] = (
            n * (1 - b) * K / 2 / a * (F * K) ** ((-b - 1) / 2) * log_FK + n / a * fk / F
        )
        risk[SabrMetrics.DSTRIKE] = (
            n * F * (1 - b) / 2 / a * log_FK * (F * K) ** ((-b - 1) / 2) - n / a * fk / K
        )
        risk[SabrMetrics.DSTRIKESTRIKE] = (
            n / a * F ** ((1 - b) / 2) * K ** ((-b - 3) / 2) * (log_FK * (b**2 - 1) / 4 + b)
        )

        return risk

    @staticmethod
    def x_risk(F, K, T, a, b, r, n) -> Dict:

        logFK = np.log(F / K)
        fk = (F * K) ** ((1 - b) / 2)
        z = n / a * fk * logFK
        dx_dz = 1 / np.sqrt(1 - 2 * r * z + z**2)

        risk = {}
        risk_z = SABRAnalytics.z_risk(F, K, T, a, b, r, n)

        risk[SabrMetrics.DALPHA] = dx_dz * risk_z[SabrMetrics.DALPHA]
        risk[SabrMetrics.DBETA] = dx_dz * risk_z[SabrMetrics.DBETA]
        risk[SabrMetrics.DRHO] = 1 / (1 - r) + (-z * dx_dz - 1) / (1 / dx_dz + z - r)
        risk[SabrMetrics.DNU] = dx_dz * risk_z[SabrMetrics.DNU]
        risk[SabrMetrics.DFORWARD] = dx_dz * risk_z[SabrMetrics.DFORWARD]
        risk[SabrMetrics.DSTRIKE] = dx_dz * risk_z[SabrMetrics.DSTRIKE]

        risk[SabrMetrics.DSTRIKESTRIKE] = (r - z) * dx_dz**3 * (
            risk_z[SabrMetrics.DSTRIKE] ** 2
        ) + dx_dz * risk_z[SabrMetrics.DSTRIKESTRIKE]

        return risk

    @staticmethod
    def C_risk(F, K, T, a, b, r, n) -> Dict:

        log_FK = np.log(F / K)
        fk = (F * K) ** ((1 - b) / 2)

        z = n / a * log_FK * fk
        risk = {}

        C0 = 1.0
        C1 = -r / 2.0
        C2 = -(r**2) / 4.0 + 1.0 / 6.0
        C3 = -(1.0 / 4.0 * r**2 - 5.0 / 24.0) * r
        C4 = -5.0 / 16.0 * r**4 + 1.0 / 3.0 * r**2 - 17.0 / 360.0
        C5 = -(7.0 / 16.0 * r**4 - 55.0 / 96.0 * r**2 + 37.0 / 240.0) * r

        dC_dz = C1 + 2 * C2 * z + 3 * C3 * z**2 + 4 * C4 * z**3 + 5 * C5 * z**4
        dC2_dz2 = 2 * C2 + 6 * C3 * z + 12 * C4 * z**2 + 20 * C5 * z**3

        risk[SabrMetrics.DRHO] = (
            -1.0 / 2 * z
            + 5.0 / 24 * z**3
            - 37.0 / 240 * z**5
            - 1.0 / 2 * z**2 * r
            + 2.0 / 3 * z**4 * r
            - 3.0 / 4 * z**3 * r**2
            + 55.0 / 32 * z**5 * r**2
            - 5.0 / 4 * z**4 * r**3
            - 35.0 / 16 * z**5 * r**4
        )
        risk_z = SABRAnalytics.z_risk(F, K, T, a, b, r, n)

        risk[SabrMetrics.DALPHA] = dC_dz * risk_z[SabrMetrics.DALPHA]
        risk[SabrMetrics.DBETA] = dC_dz * risk_z[SabrMetrics.DBETA]
        risk[SabrMetrics.DNU] = dC_dz * risk_z[SabrMetrics.DNU]
        risk[SabrMetrics.DFORWARD] = dC_dz * risk_z[SabrMetrics.DFORWARD]
        risk[SabrMetrics.DSTRIKE] = dC_dz * risk_z[SabrMetrics.DSTRIKE]
        risk[SabrMetrics.DSTRIKESTRIKE] = (
            dC_dz * risk_z[SabrMetrics.DSTRIKESTRIKE] + dC2_dz2 * risk_z[SabrMetrics.DSTRIKE] ** 2
        )
        return risk

    @staticmethod
    def _vol_and_risk(
        F, K, T, a, b, r, n, calc_risk=False, z_cut=1e-2
    ) -> Tuple[float, Dict[SabrMetrics, float]]:
        """
        Get analytical solution Lognormal Vol and Greeks
        """

        log_FK = np.log(F / K)
        fk = (F * K) ** ((1 - b) / 2)
        greeks: Dict[SabrMetrics, float] = {}

        z = n / a * log_FK * fk
        x = np.log((np.sqrt(1-2*r*z + z**2) + z - r)/(1-r))
        w1 = 1 + ((log_FK*(1-b)) ** 2)/24 + ((log_FK*(1-b)) ** 4)/1920
        w2 = ((a*(1-b)/fk)**2)/24 + a*b*r*n/(fk*4) + (2-3*r**2)*(n**2)/24

        if abs(z) < z_cut:
            # expansion when z is small
            # calculate vol and risk, you can use the helper functions above w2_risk, w1_risk, z_risk, x_risk, C_risk
            # to get the risk for each component and then aggregate them to get the risk for vol
            C0 = 1.0
            C1 = -r / 2.0
            C2 = -(r**2) / 4.0 + 1.0 / 6.0
            C3 = -(1.0 / 4.0 * r**2 - 5.0 / 24.0) * r
            C4 = -5.0 / 16.0 * r**4 + 1.0 / 3.0 * r**2 - 17.0 / 360.0
            C5 = -(7.0 / 16.0 * r**4 - 55.0 / 96.0 * r**2 + 37.0 / 240.0) * r
            C = C0 + C1 * z + C2 * z**2 + C3 * z**3 + C4 * z**4 + C5 * z**5

            sigma = a*(1+w2*T)*C/(fk*w1)

            if calc_risk:
                # Note: sigma = a * (1+ w2*T) * C / (fk*w1) 
                fk_risk = {
                    SabrMetrics.DBETA: -fk * np.log(F*K) / 2.0,
                    SabrMetrics.DFORWARD: (1-b) * fk / (2*F),
                    SabrMetrics.DSTRIKE: (1-b) * fk / (2*K),
                    SabrMetrics.DSTRIKESTRIKE: - (1-b) * fk / (2 * K ** 2) +  fk*((1-b)/(2*K))**2 # check this out!
                }

                w2_risk = SABRAnalytics.w2_risk(F,K,T,a,b,r,n)
                w1_risk = SABRAnalytics.w1_risk(F,K,T,a,b,r,n)
                C_risk = SABRAnalytics.C_risk(F,K,T,a,b,r,n)

                # sabr alpha/beta/nu/rho
                for partial, key in [
                    (SabrMetrics.DALPHA, SabrMetrics.D_LN_SIGMA_D_ALPHA),
                    (SabrMetrics.DBETA, SabrMetrics.D_LN_SIGMA_D_BETA),
                    (SabrMetrics.DRHO, SabrMetrics.D_LN_SIGMA_D_RHO),
                    (SabrMetrics.DNU, SabrMetrics.D_LN_SIGMA_D_NU),
                    (SabrMetrics.DFORWARD, SabrMetrics.D_LN_SIGMA_D_FORWARD),
                    (SabrMetrics.DSTRIKE, SabrMetrics.D_LN_SIGMA_D_STRIKE)
                ]:
                    greeks[key] = (
                        w2_risk.get(partial, 0.0) * sigma * T / (1 + w2*T)
                        - w1_risk.get(partial, 0.0) * sigma / w1
                        + C_risk.get(partial, 0.0) * sigma / C
                        - fk_risk.get(partial, 0.0) * sigma / fk
                    )
                    if partial == SabrMetrics.DALPHA: 
                        greeks[key] += sigma/a

                greeks[SabrMetrics.D_LN_SIGMA_D_TTE] = sigma * w2 / (1.0 + w2 * T)
                greeks[SabrMetrics.D_LN_SIGMA_D_STRIKESTRIKE] = sigma* (
                    w2_risk.get(SabrMetrics.DSTRIKESTRIKE, 0.0)* T / (1 + w2*T)
                    - w1_risk.get(SabrMetrics.DSTRIKESTRIKE, 0.0) / w1
                    + C_risk.get(SabrMetrics.DSTRIKESTRIKE, 0.0) / C
                    - fk_risk.get(SabrMetrics.DSTRIKESTRIKE, 0.0) / fk
                )
                greeks[SabrMetrics.D_LN_SIGMA_D_STRIKESTRIKE] += sigma *(
                    - (T * w2_risk.get(SabrMetrics.DSTRIKE, 0.0)/(1 + w2*T))**2
                    + (w1_risk.get(SabrMetrics.DSTRIKE, 0.0)/ w1)**2
                    - (C_risk.get(SabrMetrics.DSTRIKE, 0.0) / C) **2
                    + (fk_risk.get(SabrMetrics.DSTRIKE, 0.0) / fk) **2
                )
                greeks[SabrMetrics.D_LN_SIGMA_D_STRIKESTRIKE] += (
                    (greeks[SabrMetrics.D_LN_SIGMA_D_STRIKE] ** 2) / sigma
                )
            return sigma, greeks

        # raw SABR
        sigma = a*z*(1+w2*T)/(fk*x*w1)

        if calc_risk:
            # calculate risk for z, w1, w2, x
            # Note: sigma = a * (1+ w2*T) * z / (fk * w1 * x) 
            w2_risk = SABRAnalytics.w2_risk(F,K,T,a,b,r,n)
            w1_risk = SABRAnalytics.w1_risk(F,K,T,a,b,r,n)
            z_risk = SABRAnalytics.z_risk(F,K,T,a,b,r,n)
            x_risk = SABRAnalytics.x_risk(F,K,T,a,b,r,n)

            fk_risk = {
                SabrMetrics.DBETA: -fk * np.log(F*K) / 2.0,
                SabrMetrics.DFORWARD: (1-b) * fk / (2*F),
                SabrMetrics.DSTRIKE: (1-b) * fk / (2*K),
                SabrMetrics.DSTRIKESTRIKE: - (1-b) * fk / (2 * K ** 2) +  fk*((1-b)/(2*K))**2 # check this out!
            }

            # sabr alpha/beta/nu/rho
            for partial, key in [
                (SabrMetrics.DALPHA, SabrMetrics.D_LN_SIGMA_D_ALPHA),
                (SabrMetrics.DBETA, SabrMetrics.D_LN_SIGMA_D_BETA),
                (SabrMetrics.DRHO, SabrMetrics.D_LN_SIGMA_D_RHO),
                (SabrMetrics.DNU, SabrMetrics.D_LN_SIGMA_D_NU),
                (SabrMetrics.DFORWARD, SabrMetrics.D_LN_SIGMA_D_FORWARD),
                (SabrMetrics.DSTRIKE, SabrMetrics.D_LN_SIGMA_D_STRIKE)
            ]:
                greeks[key] = (
                    w2_risk.get(partial, 0.0) * sigma * T / (1 + w2*T)
                    + z_risk.get(partial, 0.0) * sigma / z
                    - w1_risk.get(partial, 0.0) * sigma / w1
                    - fk_risk.get(partial, 0.0) * sigma / fk
                    - x_risk.get(partial, 0.0) * sigma / x
                )
                if partial == SabrMetrics.DALPHA: 
                    greeks[key] += sigma/a

            greeks[SabrMetrics.D_LN_SIGMA_D_TTE] = sigma * w2 / (1.0 + w2 * T)
            greeks[SabrMetrics.D_LN_SIGMA_D_STRIKESTRIKE] = sigma* (
                w2_risk.get(SabrMetrics.DSTRIKESTRIKE, 0.0)* T / (1 + w2*T)
                + z_risk.get(SabrMetrics.DSTRIKESTRIKE, 0.0) / z
                - w1_risk.get(SabrMetrics.DSTRIKESTRIKE, 0.0) / w1
                - fk_risk.get(SabrMetrics.DSTRIKESTRIKE, 0.0) / fk
                - x_risk.get(SabrMetrics.DSTRIKESTRIKE, 0.0) / x
            )
            greeks[SabrMetrics.D_LN_SIGMA_D_STRIKESTRIKE] += sigma *(
                - (T * w2_risk.get(SabrMetrics.DSTRIKE, 0.0)/(1 + w2*T))**2
                - (z_risk.get(SabrMetrics.DSTRIKE, 0.0) / z) ** 2
                + (w1_risk.get(SabrMetrics.DSTRIKE, 0.0)/ w1)**2
                + (fk_risk.get(SabrMetrics.DSTRIKE, 0.0) / fk) **2
                + (x_risk.get(SabrMetrics.DSTRIKE, 0.0) / x) **2
            )
            greeks[SabrMetrics.D_LN_SIGMA_D_STRIKESTRIKE] += (
                (greeks[SabrMetrics.D_LN_SIGMA_D_STRIKE] ** 2) / sigma
            )
        return sigma, greeks

    @staticmethod
    def simulate_sabr_paths(
        forward: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        grids: List | np.ndarray, 
        num_paths: int = 10000,
        num_steps: int = 100,
        shift: Optional[float] = 0.0
    ):
        dt = time_to_expiry / num_steps
        sqrt_dt = np.sqrt(dt)

        F = np.full(num_paths, forward + shift)
        V = np.full(num_paths, alpha)

        for t in range(num_steps):
            Z1 = sqrt_dt*(np.random.normal(0, 1, size=num_paths))
            Z2 = sqrt_dt*(rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, size=num_paths))

            F = np.maximum(F + V * F ** beta * Z1, 1e-8)  # Avoid negative or zero shifted forward
            V = V * np.exp(-0.5 * nu**2 * dt + nu * Z2)

        edges = np.empty(len(grids) + 1, dtype=float)
        edges[1:-1] = 0.5 * (grids[:-1] + grids[1:])  # Midpoints of grid points
        edges[0] = grids[0] - 0.5 * (grids[1] - grids[0])
        edges[-1] = grids[-1] + 0.5 * (grids[-1] - grids[-2])
        counts, _ = np.histogram(F, bins=edges)
        widths = np.diff(edges)
        pdf = counts / (num_paths * widths)

        return F, (edges, pdf)

    # @staticmethod
    def pdf_and_cdf(
        forward: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        grids: List | np.ndarray,
        shift: Optional[float] = 0,
    ):
        pdf = [0]*len(grids)
        cdf = [0]*len(grids)
        for k in grids:
            pvs = [0, 0, 0]
            for idx, w in enumerate([-1, 0 , 1]):
                ln_vol = SABRAnalytics.lognormal_vol_from_alpha(forward, k + w*SABRAnalytics.EPSILON,time_to_expiry, alpha, beta, rho, nu, shift, calc_risk=False)[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
                pv = EuropeanOptionAnalytics.european_option_log_normal(forward + shift, k + w*SABRAnalytics.EPSILON + shift, time_to_expiry, ln_vol, CallOrPut.PUT)[SimpleMetrics.PV]
                pvs[idx] = pv
            pdf[grids.tolist().index(k)] = (pvs[0] - 2*pvs[1] + pvs[2]) / (SABRAnalytics.EPSILON**2)
            cdf[grids.tolist().index(k)] = (pvs[2] - pvs[0])/(2*SABRAnalytics.EPSILON)

        return pdf, cdf
    