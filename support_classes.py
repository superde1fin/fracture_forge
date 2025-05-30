import sys, os, shutil, functools
from mpi4py import MPI

class Constants:
    #Conversion factors to mks units
    units_data = {
            "real" : {"timestep" : 1e-15,
                      "energy": 6.947695457055374441297516267288312271199718686083983198028071333224698653506730719459304036593126727247072849888018891142624172072656767325378824257173291313104411727500039371381282691904398461785539532955055006054026541883753643778993966922818987711605731381144269367758849927645995442e-21,
                      "energy2ev": 0.043364104241800935172654104923666688995039473819239228763013350332162556388315988382210447569123185248294370331505027373546281365083600185155233166739425132030133025442062709369042531498461476512742224481567710879835653263457552357357584375574029172314129076114988888096334888426827447052,
                      "length": 1e-10,
                      "length2A": 1,
                      }, 

            "metal" : {"timestep" : 1e-12,
                       "energy": 1.602176634e-19,
                       "energy2ev": 1,
                       "length": 1e-10,
                       "length2A": 1,
                       },
            "si" : {"timestep" : 1,
                       "energy": 1,
                       "energy2ev": 1/(1.602176634e-19),
                       "length": 1,
                       "length2A": 1e-10,
                       },
            "cgs" : {"timestep" : 1,
                       "energy": 1e-7,
                       "energy2ev": 1/(1.602176634e-12),
                       "length": 0.01,
                       "length2A": 1e-8,
                       },
            "electron" : {"timestep" : 1e-15,
                       "energy": 4.35974e-18,
                       "energy2ev": 27.2114,
                       "length": 5.29177e-11,
                       "length2A": 0.529177,
                       },
            "micro" : {"timestep" : 1e-6,
                       "energy": 1e-12,
                       "energy2ev": 1/(1.602176634e-7),
                       "length": 1e-6,
                       "length2A": 1e-4,
                       },
            "nano" : {"timestep" : 1e-9,
                       "energy": 1e-18,
                       "energy2ev": 1/(1.602176634e-1),
                       "length": 1e-9,
                       "length2A": 1e-1,
                       },
                 }
    boltzman = 1.3806452e-23 #J/K


class SystemDefaults:
    simulation_temp = 300
    spanning_radius = 1
    error = 0.1
    units = "real"
    max_verbosity = 5
    verbosity = 1
    structure_file = "STRUCTURE"
    forcefield_file  = "FORCEFIELD"
    action_proc = 0
    direction = 2
    plane = 3

class RuntimeData:
    old_boudns = None #List of size 2 with minimum and maximum atom y coordinates
    structure_file = None #The atom arrangement for the path search to be executed on
    forcefield_file = None #File outlining masses, charges, and pairwise potential eneries
    verbosity = None #Integer value of verbosity set by user
    untis = None #A string representing the name of lammps unit set
    minimize = None #A boolean switch to enable stepwise minimization
    simulation_temp = None #Temperature of the simulation selected by the user

    #Condition that gets turned on when maximum verbosity is selected
    #Creates a log directory and stores per-node LAMMPS logs
    test_mode = None 
    direction = None #Direction of initial fracture propagation
    plane = None #Plane that the incremental cuts are made across

class Helper:
    action_proc = SystemDefaults.action_proc
    @staticmethod
    def mpi_print(*args, verbosity = None):
        out_str = ""

        if MPI.COMM_WORLD.Get_rank() == Helper.action_proc:
            if verbosity is None:
                verbosity = SystemDefaults.verbosity
            if verbosity <= RuntimeData.verbosity:
                print(*args)
                out_str = " ".join(map(str, args)) + "\n"
                sys.stdout.flush()

        return out_str

    @staticmethod
    def convert_timestep(lmp, step): #ns  - step
        return int((step*1e-9)/(lmp.eval("dt")*Data.units_data[SystemParams.parameters["units"]]["timestep"]))

    @staticmethod
    def action(command, *args, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == Helper.action_proc:
            return command(*args, **kwargs)
        else:
            return None

    @staticmethod
    def linear_func(func):
        @functools.wraps(func)
        def decorator(*args, **kwargs):
            if MPI.COMM_WORLD.Get_rank() == 0:
                return func(*args, **kwargs)
            else:
                return None
        return decorator
