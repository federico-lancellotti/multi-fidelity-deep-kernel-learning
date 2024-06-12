import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.fftpack import fft2, ifft2
from abc import ABC, abstractmethod


class PDESolver(ABC):
    """
    Abstract class for solving partial differential equations.

    Attributes:
        d (float): The diffusion coefficient.
        mu (float): The reaction coefficient.
        T (float): The final time.
        dt (float): The time step.
        L (float): The size of the domain.
        n (int): The number of grid points in one dimension.
        N (int): The total number of grid points.

    Methods:
        solve: Solves the partial differential equation.
        _rhs: Computes the right-hand side of the partial differential equation.
    """

    @abstractmethod
    def __init__(self, d, mu, T, dt, L, n):
        self.d = d
        self.mu = mu
        self.T = T
        self.dt = dt
        self.L = L
        self.n = n
        self.N = n * n

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def _rhs(self, *args):
        pass


class ReactionDiffusionSolver(PDESolver):
    """
    Class for solving the reaction-diffusion system.
    Inherits from the abstract class PDESolver.

    Attributes:
        d (float): The diffusion coefficient.
        mu (float): The reaction coefficient.
        T (float): The final time.
        dt (float): The time step.
        L (float): The size of the domain.
        n (int): The number of grid points in one dimension.
        N (int): The total number of grid points.

    Methods:
        solve: Solves the reaction-diffusion system.
        _rhs: Computes the right-hand side of the reaction-diffusion system.
    """

    def __init__(self, d, mu, T, dt, L, n):
        super().__init__(d=d, mu=mu, T=T, dt=dt, L=L, n=n)


    def solve(self):
        """
        Solves the reaction-diffusion system for the given level of fidelity and parameter mu.

        Args:
            None.

        Returns:
            The solution of the reaction-diffusion system.
        """

        # Parameters
        t_span = (0, self.T)
        t_eval = np.arange(0, self.T, self.dt)

        # Set mesh and wave numbers
        x2 = np.linspace(-self.L/2, self.L/2, self.n+1)
        x = x2[:-1]
        y = x
        kx = (2 * np.pi / self.L) * np.concatenate([np.arange(0, self.n//2), np.arange(-self.n//2, 0)])
        ky = kx

        X, Y = np.meshgrid(x, y)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K22 = K2.flatten()

        # Initial conditions
        u_ini = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
        v_ini = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))

        uvt = np.concatenate([fft2(u_ini).flatten(), fft2(v_ini).flatten()])

        # Solve the reaction-diffusion system
        sol = solve_ivp(
            self._rhs, t_span, uvt, t_eval=t_eval, 
            args=(K22, self.d, self.mu, self.n, self.N), 
            method='RK45')

        # Initialize the tensors to store the results
        u = np.zeros((self.n, self.n, len(sol.t)))
        v = np.zeros((self.n, self.n, len(sol.t)))

        # Store the results in the tensors
        for t in range(len(sol.t)):
            u_t = np.real(ifft2(np.reshape(sol.y[:self.N, t], (self.n, self.n))))
            v_t = np.real(ifft2(np.reshape(sol.y[self.N:, t], (self.n, self.n))))
            u[:, :, t] = u_t
            u[:, :, t] = v_t

        # return u, v
        return u


    def _rhs(self, t, uvt, K22, d, mu, n, N):
        """
        Computes the right-hand side of the reaction-diffusion system.

        Args:
            t: Time.
            uvt: The current state of the system.
            K22: The squared wavenumber.
            d: The diffusion coefficient.
            mu: The reaction coefficient.
            n: The number of grid points in one dimension.
            N: The total number of grid points.

        Returns:
            The right-hand side of the reaction-diffusion system.
        """

        ut = np.reshape(uvt[:N], (n, n))
        vt = np.reshape(uvt[N:], (n, n))
        u = np.real(ifft2(ut))
        v = np.real(ifft2(vt))

        u3 = u**3
        v3 = v**3
        u2v = (u**2) * v
        uv2 = u * (v**2)
        utrhs = np.reshape(fft2(u - u3 - uv2 + mu * u2v + mu * v3), N)
        vtrhs = np.reshape(fft2(v - u2v - v3 - mu * u3 - mu * uv2), N)

        rhs = np.concatenate([
            -d * K22 * uvt[:N] + utrhs,
            -d * K22 * uvt[N:] + vtrhs
        ])
        
        return rhs


class DiffusionAdvectionSolver(PDESolver):
    """
    Class for solving the diffusion-advection system.
    Inherits from the abstract class PDESolver.

    Attributes:
        d (float): The diffusion coefficient.
        mu (float): The transport coefficient.
        T (float): The final time.
        dt (float): The time step.
        L (float): The size of the domain.
        n (int): The number of grid points in one dimension.
        N (int): The total number of grid points.

    Methods:
        solve: Solves the diffusion-advection system.
        _rhs: Computes the right-hand side of the diffusion-advection system.
    """

    def __init__(self, d=0.001, mu=3, T=20, dt=0.25, L=5, n=200):
        super().__init__(d=d, mu=mu, T=T, dt=dt, L=L, n=n)


    def solve(self):
        """
        Solves the diffusion-transport system for the given level of fidelity and parameter mu.

        Args:
            None.

        Returns:
            The solution of the diffusion-transport system.
        """

        # Parameters
        t_eval = np.arange(0, self.T, self.dt)

        # Set mesh and wave numbers
        x2 = np.linspace(-self.L, self.L, self.n+1)
        x = x2[:self.n]
        y = x
        # kx = (2*np.pi / self.L) * np.fft.fftfreq(self.n, d=2*self.L/self.n)
        kx = (2 * np.pi / self.L) * np.concatenate([np.arange(0, self.n/2), np.arange(-self.n/2, 0)])
        ky = kx

        X, Y = np.meshgrid(x, y)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K22 = K2.reshape(self.N, 1)

        # Initial conditions
        omega_ini = np.exp(-2 * X**2 - Y**2 / 20)
        omega_hat_ini = np.fft.fft2(omega_ini).reshape(1, self.N).conj().real

        # Solve the diffusion-transport system
        omega_hat_sol = odeint(self._rhs, omega_hat_ini.flatten(), t_eval, 
                               args=(KX, KY, K22, self.d, self.mu, self.n, self.N))

        # Initialize the tensor to store the results
        omega = np.zeros((self.n, self.n, len(t_eval)))

        # Inverse Fourier transform of the vorticity to obtain the solution
        for j in range(len(t_eval)):
            #omega[:, :, j] = np.real(ifft2(omega_hat_sol[j, :].reshape((self.n, self.n))))
            omega[:, :, j] = np.fft.ifft2(omega_hat_sol[j, :].reshape(self.n, self.n)).real

        return omega


    def _rhs(self, omega_hat, t, KX, KY, K22, d, mu, n, N):   
        """
        Right-hand side of the diffusion-transport system.

        Args:
            omega_hat (ndarray): Flattened vorticity.
            t (float): Time.
            KX (ndarray): Wave numbers in the x-direction.
            KY (ndarray): Wave numbers in the y-direction.
            K22 (ndarray): Wave numbers squared (of shape (N,1)).
            d (float): Diffusion coefficient.
            mu (float): Transport coefficient.
            n (int): Grid size.
            N (int): Number of grid points.

        Returns:
            ndarray: Right-hand side of the system.
        """

        # Reshape the vorticity
        omega_hat = omega_hat.reshape(N, 1)

        # Inverse Fourier transform of the vorticity
        omega = np.fft.ifft2(omega_hat.reshape((n, n)))
        
        # Poisson equation for the stream function
        K22[K22==0] = 1e-10
        psi_hat = -omega_hat / K22
        psi_hat[0] = 0  # to handle the case K=0
        psi = np.fft.ifft2(psi_hat.reshape((n, n)))
        
        # Compute the gradients
        psi_x = np.fft.ifft2(1j * KX * psi_hat.reshape((n, n)))
        psi_y = np.fft.ifft2(1j * KY * psi_hat.reshape((n, n)))
        omega_x = np.fft.ifft2(1j * KX * omega_hat.reshape((n, n)))
        omega_y = np.fft.ifft2(1j * KY * omega_hat.reshape((n, n)))

        # Non linear term
        non_linear = mu * (psi_x * omega_y - psi_y * omega_x)
        
        # Fourier transform of the non linear term
        non_linear_hat = np.fft.fft2(non_linear).reshape(N, 1)
        
        # Right-hand side of the system
        rhs = -non_linear_hat - d * K22 * omega_hat
        rhs = np.real(rhs.flatten())

        return rhs