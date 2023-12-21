"""gen_geo_brownian.py"""

from typing import Optional, Union

import numpy as np

import CIR_proc
import OU_proc

# =============================================================================
# mu, OU drift process

# ---------------------------------------------------
# Drift Protocol implementation


class OUDrift:
    """
    Implements the geometric_brownian.Drift Protocol.
    Ornstein-Uhlenbeck drift processes matrix, each column of the
    drift matrix is 1D a process.
    - T is the sample size of the processes.
    - OU_params can be a an instance of OU_proc.OUParams, in that case
        all processes have the same parameters. It can also be a tuple,
        in that case each process will have the parameters in the tuple,
        each column in the resulting 2D array corresponds to the tuple index.
    - n_procs is ignored if OU_proc.OU_params is tuple, else, corresponds to
        the number of processes desired. If OU_params is not tuple and
        n_procs is None, will raise ValueError.
    - rho is the correlation coefficient.
    """

    def __init__(
        self,
        T: int,
        OU_params: Union[OU_proc.OUParams, tuple[OU_proc.OUParams, ...]],
        n_procs: Optional[int] = None,
        rho: Optional[float] = None,
    ) -> None:
        self.T = T
        self.OU_params = OU_params
        self._n_procs = n_procs
        self.rho = rho

        self._n_procs_ = self._get_n_procs()

    @property
    def sample_size(self) -> int:
        """The sample size T of the drift processes."""
        return self.T

    @property
    def n_procs(self) -> int:
        """The number of drift processes."""
        return self._n_procs_

    def get_mu(self, random_state: Optional[int] = None) -> np.ndarray:
        """
        Returns a 2D array, each column is a 1D process,
        the number of rows define the sample size,
        random_state not None to reproduce results.
        """
        return OU_proc.get_corr_OU_procs(
            self.T, self.OU_params, self._n_procs_, self.rho, random_state
        )

    def _get_n_procs(self) -> int:
        """
        Check what is the correct n_proc, depening on params type and
        n_procs type.
        - if params is instance of OU_proc.OUParams then n_procs cannot be None,
            the value of n_procs is then passed as the correct one.
        - if params is tuple of OU_proc.OUParams n_procs input argument is
            ignored and the correct n_procs is then the size of the tuple.
        """
        if isinstance(self.OU_params, tuple):
            return len(self.OU_params)
        elif self._n_procs is None:
            raise ValueError(
                "If OU_params is not tuple, n_procs cannot be None."
            )
        return self._n_procs


# ---------------------------------------------------
# parameter estimation


def estimate_drift_OU_params(
    proc_mat: np.ndarray, rolling_window: int
) -> tuple[OU_proc.OUParams, ...]:
    """
    Estimate drift OU_proc.OUParams from data
    (gneralized geometric Brownian motion paths).
    Rolls the the diffusion increments to generate new processes
    for each mu. Then estimate its OU_proc.OUParams
    - proc_mat is a 2D array,  each column is a process
    - rolling window defines the lenght of the window to calculate
        the mean of the increments.
    Returns a tuple of OU_proc.OUParams indexed with the same order as proc_mat
    columns.
    """
    diffusion_increments = np.diff(proc_mat, axis=0) / proc_mat[:-1, :]
    rolled_increments = np.lib.stride_tricks.sliding_window_view(
        diffusion_increments, rolling_window, axis=0
    )
    rolling_mus = np.mean(rolled_increments, axis=-1)
    return tuple(
        [
            OU_proc.estimate_OU_params(rolling_mus[:, i])
            for i in range(rolling_mus.shape[1])
        ]
    )


# ---------------------------------------------------
# drift correlation


def estimate_drift_correlation(
    proc_mat: np.ndarray, rolling_window: int
) -> float:
    """
    Estimate drift processes mean correlation from data
    (geometric Brownian motion paths).
    - proc_mat is a 2D array,  each column is a process
    - rolling window defines the lenght of the window to calculate
        the mean of the increments.
    Returns a float, the correlation coefficient.
    """
    diffusion_increments = np.diff(proc_mat, axis=0) / proc_mat[:-1, :]
    rolled_increments = np.lib.stride_tricks.sliding_window_view(
        diffusion_increments, rolling_window, axis=0
    )
    rolling_mus = np.mean(rolled_increments, axis=-1)
    corr_mat = np.corrcoef(rolling_mus, rowvar=False)
    # put nan in correlation matrix diagonal to exclude it
    # when taking the mean (nanmean)
    np.fill_diagonal(corr_mat, np.nan)
    return float(np.nanmean(corr_mat))


# =============================================================================
# sigma, CIR process

# ---------------------------------------------------
# Sigma Protocol implementation


class CIRSigma:
    """
    Implements the geometric_brownian.Sigma Protocol.
    Cox-Ingersoll-Ross sigma processes matrix, each column of the
    drift matrix is 1D a process.
    - T is the sample size of the processes.
    - CIR_params can be a an instance of CIR_proc.CIRParams, in that case
        all processes have the same parameters. It can also be a tuple,
        in that case each process will have the parameters in the tuple,
        each column in the resulting 2D array corresponds to the tuple index.
    - n_procs is ignored if CIR_proc.CIR_params is tuple, else, corresponds to
        the number of processes desired. If CIR_params is not tuple and
        n_procs is None, will raise ValueError.
    - rho is the correlation coefficient.
    """

    def __init__(
        self,
        T: int,
        CIR_params: Union[CIR_proc.CIRParams, tuple[CIR_proc.CIRParams, ...]],
        n_procs: Optional[int] = None,
        rho: Optional[float] = None,
    ) -> None:
        self.T = T
        self.CIR_params = CIR_params
        self._n_procs = n_procs
        self.rho = rho

        self._n_procs_ = self._get_n_procs()

    @property
    def sample_size(self) -> int:
        """The sample size T of the drift processes."""
        return self.T

    @property
    def n_procs(self) -> int:
        """The number of drift processes."""
        return self._n_procs_

    def get_sigma(self, random_state: Optional[int] = None) -> np.ndarray:
        """
        Returns a 2D array, each column is a 1D process,
        the number of rows define the sample size,
        random_state not None to reproduce results.
        """
        return CIR_proc.get_corr_CIR_procs(
            self.T, self.CIR_params, self._n_procs_, self.rho, random_state
        )

    def _get_n_procs(self) -> int:
        """
        Check what is the correct n_proc, depening on params type and
        n_procs type.
        - if params is instance of OU_proc.OUParams then n_procs cannot be None,
            the value of n_procs is then passed as the correct one.
        - if params is tuple of OU_proc.OUParams n_procs input argument is
            ignored and the correct n_procs is then the size of the tuple.
        """
        if isinstance(self.CIR_params, tuple):
            return len(self.CIR_params)
        elif self._n_procs is None:
            raise ValueError(
                "If CIR_params is not tuple, n_procs cannot be None."
            )
        return self._n_procs


# ---------------------------------------------------
# parameter estimation


def estimate_sigma_CIR_params(
    proc_mat: np.ndarray, rolling_window: int
) -> tuple[CIR_proc.CIRParams, ...]:
    """
    Estimate drift CIR_proc.CIRParams from data
    (gneralized geometric Brownian motion paths).
    Rolls the the diffusion increments to generate new processes
    for each mu. Then estimate its CIR_proc.CIRParams
    - proc_mat is a 2D array,  each column is a process
    - rolling window defines the lenght of the window to calculate
        the std of the increments.
    Returns a tuple of CIR_proc.CIRParams indexed with the same order as proc_mat
    columns.
    """
    diffusion_increments = np.diff(proc_mat, axis=0) / proc_mat[:-1, :]
    rolled_increments = np.lib.stride_tricks.sliding_window_view(
        diffusion_increments, rolling_window, axis=0
    )
    rolling_sigmas = np.std(rolled_increments, axis=-1)
    return tuple(
        [
            CIR_proc.estimate_CIR_params(rolling_sigmas[:, i])
            for i in range(rolling_sigmas.shape[1])
        ]
    )


# ---------------------------------------------------
# sigma correlation


def estimate_sigma_correlation(
    proc_mat: np.ndarray, rolling_window: int
) -> float:
    """
    Estimate sigma processes mean correlation from data
    (geometric Brownian motion paths).
    - proc_mat is a 2D array,  each column is a process
    - rolling window defines the lenght of the window to calculate
        the std of the increments.
    Returns a float, the correlation coefficient.
    """
    diffusion_increments = np.diff(proc_mat, axis=0) / proc_mat[:-1, :]
    rolled_increments = np.lib.stride_tricks.sliding_window_view(
        diffusion_increments, rolling_window, axis=0
    )
    rolling_sigmas = np.std(rolled_increments, axis=-1)
    corr_mat = np.corrcoef(rolling_sigmas, rowvar=False)
    # put nan in correlation matrix diagonal to exclude it
    # when taking the mean (nanmean)
    np.fill_diagonal(corr_mat, np.nan)
    return float(np.nanmean(corr_mat))