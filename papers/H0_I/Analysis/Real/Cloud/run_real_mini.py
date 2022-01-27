""" Run mini Cube for the Real data in the Cloud """

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse
import numpy as np
import os, sys

from concurrent.futures import ProcessPoolExecutor
import subprocess

from zdm import iteration as it
from zdm import io

from IPython import embed

def main(pargs, pfile:str, oproot:str, outdir:str='Output'):

    # Generate the folder?
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    ############## Load up ##############
    input_dict=io.process_jfile(pfile)

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    npoints = np.array([item['n'] for key, item in vparam_dict.items()])
    ntotal = int(np.prod(np.abs(npoints)))

    # Total number of CPUs to be running on this Cube
    total_ncpu = pargs.ncpu if pargs.total_ncpu is None else pargs.total_ncpu
    batch = 1 if pargs.batch is None else pargs.batch

    nper_cpu = ntotal // total_ncpu
    if int(ntotal/total_ncpu) != nper_cpu:
        raise IOError(f"Ncpu={total_ncpu} must divide evenly into ntotal={ntotal}")

    commands = []
    for kk in range(pargs.ncpu):
        line = []
        # Which CPU is running out of the total?
        iCPU = (batch-1)*pargs.ncpu + kk
        outfile = os.path.join(outdir, oproot.replace('.out', f'{iCPU+1}.out'))
        # Command
        line = ['python', 
                '../py/build_real_cube.py', 
                '-n', f'{iCPU+1}',
                '-m', f'{nper_cpu}', 
                '-o', f'{outfile}',
                '--clobber',
                '-p', f'{pfile}']
        # Finish
        commands.append(line)

    # Launch em!
    processes = []
    for command in commands:
        # Popen
        print(f"Running this command: {' '.join(command)}")
        pw = subprocess.Popen(command)
        processes.append(pw)

    # Wait on em!
    for pw in processes:
        pw.wait()

    print("All done!")

def parse_option():
    # test for command-line arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--ncpu',type=int, required=True,help="Number of CPUs to run on (might be split in batches)")
    parser.add_argument('-t','--total_ncpu',type=int, required=False,help="Total number of CPUs to run on (might be split in batches)")
    parser.add_argument('-b','--batch',type=int, default=1, required=False,help="Batch number")
    #parser.add_argument('--NFRB',type=int,required=False,help="Number of FRBs to analzye")
    #parser.add_argument('--iFRB',type=int,default=0,help="Initial FRB to run from")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # get the argument of training.
    pfile = '../Cubes/real_mini_cube.json'
    oproot = 'real_mini.out' 
    pargs = parse_option()
    main(pargs, pfile, oproot)
