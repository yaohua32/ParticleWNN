# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:07:03 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:07:03 
#  */
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
#
try:
    import scienceplots
except:
    pass
#
plt.rcParams['axes.titlesize'] = 22 
plt.rcParams['axes.labelsize'] = 22  
plt.rcParams['xtick.labelsize'] = 18  
plt.rcParams['ytick.labelsize'] = 18  
plt.rcParams['legend.fontsize'] = 22

class Draw():

    def __init__(self):
        pass

    def show_1d_list(self, x_list:list, y_list:list, label_list:list, 
                     y_name:str=None, 
                     save_path:str=None)->None:
        '''
        '''
        x_plot = np.linspace(-1., 1., 1000)
        #
        plt.figure(figsize=(8,5))
        if type(x_list).__name__!='list':
            x_list = [x_list] * len(y_list)
        #
        with plt.style.context(['science', 'no-latex']):
            for x,y,label in zip(x_list, y_list, label_list):
                y_plot = griddata(x.flatten(), y.flatten(), x_plot, method='cubic')
                plt.plot(x_plot, y_plot, '-.', linewidth=3., label=label)
            #
            plt.xlabel('x')
            if y_name is not None:
                plt.ylabel(y_name)
            plt.tight_layout()
            plt.legend()
        #
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def show_2d(self, x, y, title=None, points_list=None, save_path=None):
        '''
        '''
        mesh = np.meshgrid(np.linspace(-1., 1., 200), np.linspace(-1., 1.,200))
        x_plot, y_plot = mesh[0], mesh[1]
        #
        with plt.style.context(['science', 'no-latex']):
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
            #
            z_plot = griddata((x[:,0], x[:,1]), np.ravel(y), (x_plot, y_plot), method='linear')
            cntr = axs.contourf(x_plot, y_plot, z_plot, levels=14, cmap='RdBu_r')
            fig.colorbar(cntr, ax=axs)
            #
            if points_list is not None:
                for points in points_list:
                    plt.scatter(points[:,0], points[:,1], s=20, lw=2., c='k', marker='o')
            #
            if title is not None:
                axs.set_title(title)
            axs.set_xlabel('x')
            axs.set_ylabel('y')
            plt.tight_layout()
        #
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def show_error_2d(self, x_list, y_list, label_list, 
                      title_list:str=None,
                      save_path=None):
        '''
        '''
        mesh = np.meshgrid(np.linspace(-1., 1., 200), np.linspace(-1., 1.,200))
        x_plot, y_plot = mesh[0], mesh[1]
        #
        ncols = len(x_list)
        plt.figure(figsize=(7*ncols,5))
        #
        with plt.style.context(['science', 'no-latex']):
            n_fig = 1
            for x, y, label in zip(x_list, y_list, label_list):
                plt.subplot(1, ncols, n_fig)
                z_plot = griddata((x[:,0], x[:,1]), np.ravel(y), (x_plot, y_plot), method='linear')
                cntr = plt.contourf(x_plot, y_plot, z_plot, levels=14, cmap='RdBu_r')
                plt.colorbar(cntr, pad=0.05, aspect=10)
                #
                if title_list is not None:
                    plt.title(title_list[n_fig-1])
                plt.xlabel('x')
                plt.ylabel('y')
                #
                n_fig+=1
        #
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def show_tRely_1d_list(self, t_list, x_list, u_list, 
                           label_list:str, save_path=None):
        '''
        '''
        #
        mesh = np.meshgrid(np.linspace(0., 1., 100), np.linspace(-1., 1.,200))
        x_plot, y_plot = mesh[0], mesh[1]
        #
        with plt.style.context(['science', 'no-latex']):
            plt.figure(figsize=(10,8))
            level = 24
            #
            nrow = len(u_list)
            for i in range(nrow):
                t, x, u, label = t_list[i], x_list[i], u_list[i], label_list[i]
                #
                plt.subplot(nrow, 1, i+1)
                z_plot = griddata((t.flatten(), x.flatten()), np.ravel(u), (x_plot, y_plot), method='linear')
                cntr = plt.contourf(x_plot, y_plot, z_plot, cmap='rainbow', levels=level)
                plt.colorbar(cntr, pad=0.05, aspect=3)
                #
                plt.title(f'{label}')
                plt.ylabel('x')
            plt.xlabel('t') 
            plt.tight_layout()
        #
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def show_confid(self, times_list, errors_list, name_list:list,
                    x_name:str, y_name:str, 
                    save_path=None, confid=True):
        '''
        '''
        with plt.style.context(['science', 'no-latex']):
            plt.figure(figsize=(7,5))
            for times, errors, name in zip(times_list, errors_list, name_list):
                err_avg = np.mean(errors, axis=0)
                time_avg = np.mean(times, axis=0)
                low_bd = err_avg - 1.96 * np.std(errors, axis=0) / np.sqrt(len(err_avg))
                high_bd = err_avg + 1.96 * np.std(errors, axis=0) / np.sqrt(len(err_avg))
                plt.plot(time_avg, err_avg, linewidth=4., label=name)
                if confid:
                    plt.fill_between(time_avg, low_bd, high_bd, alpha=0.5)
            plt.yscale('log')
            plt.grid(linestyle='-.')
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            #
            plt.tight_layout()
            plt.legend()
        #
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

if __name__=='__main__':
    draw = Draw()
