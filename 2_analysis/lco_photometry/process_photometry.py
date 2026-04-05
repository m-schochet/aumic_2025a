"""Script to run all jobs in this folder"""
import argparse
import subprocess
import sys
import os


def run_scripts_sequentially(script_list, args):
    """This function loops through thee `scripts_to_run` variable
    from below to execute and print outputs from each script"""

    for index, script in enumerate(script_list):
        if args[index]=='y':
            print(f"--- Running {script} ---")
            try:
                # The command is a list: ['python', 'script_name.py']
                # Use sys.executable to ensure the current Python interpreter is used
                result = subprocess.run([sys.executable, os.path.join("scripts_photometry", script)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.stdout is not None:
                    print(f"Output from {script}:\n{result.stdout}")
                print(f"--- Finished {script} successfully ---")

            except subprocess.CalledProcessError as e:
                print(f"!!! Error running {script} !!!")
                print(f"Stderr:\n{e.stderr}")
                print(f"Stdout:\n{e.stdout}")

                # Stop execution if any script fails
                sys.exit(1)

            except FileNotFoundError:
                print(f"!!! Error: {script} not found. Check path. !!!")
                sys.exit(1)
            
        elif args[index]=='n':
            print(f"--- Skipping {script} ---")

        else:
            print(f"--- Invalid option for {script} argument ---")

if __name__ == "__main__":
    scripts_to_run = [
        "badfiles_sinistro.py", #e1
        "ogsinistro.py", #e2
        "sinistrocompleteness.py", #e3
        "muscat.py", #e4
        "binsinistro.py", #e5
        "cheops.py", #e6
        "gen_fig1.py" #e7
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-e1', '--exec1', default='y', type = str, \
                        help = 'Do you want identify non-used files for Sinistro light curves (y/n). Default is y')
    
    parser.add_argument('-e2', '--exec2', default='y', type = str, \
                        help = 'Do you want plot the un-processed light curves for Sinistro (y/n). Default is y')
    
    parser.add_argument('-e3', '--exec3', default='y', type = str, \
                        help = 'Do you want identify the completeness percentage (used/taken) of \
                            Sinistro light curves (y/n). Default is y')
    
    parser.add_argument('-e4', '--exec4', default='y', type = str, \
                        help = 'Do you want plot/save the light curves for MuSCAT (y/n). Default is y')
    
    parser.add_argument('-e5', '--exec5', default='y', type = str, \
                        help = 'Do you want bin the Sinistro light curves to increase SNR. Default is y')
    
    parser.add_argument('-e6', '--exec6', default='y', type = str, \
                        help = 'Do you want to plot/save the CHEOPS light curves (y/n). Default is y')
    
    parser.add_argument('-e7', '--exec7', default='y', type = str, \
                        help = 'Do you want generate figure 1 (y/n). NOTE: must \
                            run e5 and the `1_dataprep/twirler.py` script. Default is y')

    args = parser.parse_args()
    argument_values = [args.exec1, args.exec2, args.exec3, args.exec4, args.exec5, args.exec6, args.exec7]

    run_scripts_sequentially(scripts_to_run, argument_values)
    