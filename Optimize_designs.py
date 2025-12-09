import time
import matplotlib.pyplot as plt
import os
import numpy as np

def plotting_log_file(log_file_name):
    
    weights = np.load(log_file_name, allow_pickle=True)
    
    (size, ) = weights.shape
    
    param_history = weights[0]

    loss_function = weights[1]
    t_elapsed = weights[2]
    
    # Directionality and Quality factor are in thruples
    resonance_wavelength = weights[3]
    quality_factor = weights[4]
    
    epochs = np.arange(len(loss_function)) + 1
    
    fig = plt.figure(figsize=(15, 8))
    spec = fig.add_gridspec(ncols=1, nrows=size-1) # subplot grid
        
    plt.rcParams.update({'font.size': 14})  # Change default font size for text

    fig.add_subplot(spec[0, 0])
    plt.plot(epochs, quality_factor)
    plt.yscale('log')
    plt.ylim([0.9*10**int(np.log10(np.min(quality_factor))), 
              1.1*10**int(np.log10(np.max(quality_factor))+1)])
    plt.grid(True)
    plt.xticks([])
    plt.ylabel('Quality Factor')

    fig.add_subplot(spec[1, 0])
    plt.plot(epochs, resonance_wavelength)
    plt.grid(True)
    plt.xticks([])
    plt.ylabel('Wavelength')
    
    fig.add_subplot(spec[2, 0])
    plt.plot(epochs, t_elapsed)
    plt.ylabel('time/epoch (s)')
    plt.xticks([])
    plt.grid(True)
    
    fig.add_subplot(spec[3, 0])
    plt.plot(epochs, loss_function)
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.ylim([10**int(np.log10(np.min(loss_function))-1), 2])
    plt.xlabel('Epochs')
        
    head, tail = os.path.split(log_file_name)
    plt.suptitle(tail[:-4])

    fig_save_file = f'./{head}/plots/' + tail[:-4] + '.png'
    fig.savefig(fig_save_file)
    
    param_save_file = f'./{head}/param_history/' + tail[:-4] + '_param_history.npy'
    np.save(param_save_file, param_history)
    
    plt.show()


class Optimize:
    def __init__(
        self,
        objective_function,
    ):
        self.objective = objective_function

        # Some internal variables
        self.iteration = 0
        self.of_list = []
        self.p_opt = []
        self.t_store = time.time()
        self.param_history = np.array([])
        self.t_elapsed_array = np.array([])

    def _parse_bounds(self, bounds):
        """Parse the input bounds, which can be 'None', a list with two
        elements, or a list of tuples with 2 elements each
        """
        try:
            if bounds is None:
                return None
            elif not isinstance(bounds[0], tuple):
                if len(bounds) == 2:
                    return [tuple(bounds) for i in range(self.params.size)]
                else:
                    raise ValueError
            elif len(bounds) == self.params.size:
                if all([len(b) == 2 for b in bounds]):
                    return bounds
                else:
                    raise ValueError
            else:
                raise ValueError
        except:
            raise ValueError(
                "'bounds' should be a list of two elements "
                "[lb, ub], or a list of the same length as the number of "
                "parameters where each element is a tuple (lb, ub)"
            )

    def _disp(self, t_elapsed):
        """Display information at every iteration"""
        disp_str = "Epoch: %4d/%4d | Duration: %6.2f secs" % (
            self.iteration,
            self.Nepochs,
            t_elapsed,
        )
        disp_str += " | Objective: %4e" % self.of_list[-1]
        if self.disp_p:
            disp_str += " | Parameters: %s" % self.params
        print(disp_str)

    def adam(
        self,
        pstart,
        Nepochs=50,
        bounds=None,
        disp_p=False,
        step_size=1e-2,
        beta1=0.9,
        beta2=0.999,
        args=(),
        pass_self=False,
        callback=None,
    ):
        """Performs 'Nepoch' steps of ADAM minimization with parameters
        'step_size', 'beta1', 'beta2'

        Additional arguments:
        bounds          -- can be 'None', a list of two elements, or a
            scipy.minimize-like list of tuples each containing two elements
            The 'bounds' are set abruptly after the update step by snapping the
            parameters that lie outside to the bounds value
        disp_p          -- if True, the current parameters are displayed at
            every iteration
        args            -- extra arguments passed to the objective function
        pass_self       -- if True, then the objective function should take
            of(params, args, opt), where opt is an instance of the Minimize
            class defined here. Useful for scheduling
        Callback        -- function to call at every epoch; the argument that's
                            passed in is the current minimizer state
        """
        self.params = pstart
        self.bounds = self._parse_bounds(bounds)
        self.Nepochs = Nepochs
        self.disp_p = disp_p

        # Restart the counters
        self.iteration = 0
        self.t_store = time.time()
        self.of_list = []

        if pass_self:
            arglist = list(args)
            arglist.append(self)
            args = tuple(arglist)

        # def adam_call(iteration):

        for iteration in range(Nepochs):
            self.iteration += 1

            self.t_store = time.time()
            of, grad = self.objective(self.params, args)

            t_elapsed = time.time() - self.t_store
            self.t_elapsed_array = np.append(self.t_elapsed_array, t_elapsed)

            self.of_list.append(of)
            self._disp(t_elapsed)

            if iteration == 0:
                mopt = np.zeros(grad.shape)
                vopt = np.zeros(grad.shape)

            (grad_adam, mopt, vopt) = self._step_adam(
                grad, mopt, vopt, iteration, beta1, beta2
            )
            # Change parameters towards minimizing the objective
            if iteration < Nepochs - 1:
                self.params = (
                    self.params
                    - step_size * np.exp(-1 * iteration / Nepochs) * grad_adam
                )

            self.param_history = np.append(self.param_history, np.array(self.params))

            if bounds:
                lbs = np.array([b[0] for b in self.bounds])
                ubs = np.array([b[1] for b in self.bounds])
                self.params[self.params < lbs] = lbs[self.params < lbs]
                self.params[self.params > ubs] = ubs[self.params > ubs]

            if callback is not None:
                callback(self)

        return (self.params, self.of_list, self.param_history, self.t_elapsed_array)

    @staticmethod
    def _step_adam(gradient, mopt_old, vopt_old, iteration, beta1, beta2, epsilon=1e-8):
        """Performs one step of Adam optimization"""

        mopt = beta1 * mopt_old + (1 - beta1) * gradient
        mopt_t = mopt / (1 - beta1 ** (iteration + 1))
        vopt = beta2 * vopt_old + (1 - beta2) * (np.square(gradient))
        vopt_t = vopt / (1 - beta2 ** (iteration + 1))
        grad_adam = mopt_t / (np.sqrt(vopt_t) + epsilon)

        return (grad_adam, mopt, vopt)
