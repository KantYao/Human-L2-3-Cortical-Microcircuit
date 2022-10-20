#===========================================================================
# Import, Set up MPI Variables, Load Necessary Files
#===========================================================================
from mpi4py import MPI
import time
tic_0 = time.perf_counter() #script runtime calculation value
import os
from os.path import join
import sys
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy import stats as st
import neuron
from neuron import h, gui
import LFPy
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, StimIntElectrode
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import pandas as pd

#MPI variables:
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
GLOBALSEED = int(sys.argv[1])

# Create new RandomState for each RANK
SEED = GLOBALSEED*10000
np.random.seed(SEED + RANK)
local_state = np.random.RandomState(SEED + RANK)
halfnorm_rv = st.halfnorm
halfnorm_rv.random_state = local_state
uniform_rv = st.uniform
uniform_rv.random_state = local_state
#from net_params import *

#Mechanisms and files
print('Mechanisms found: ', os.path.isfile('mod/x86_64/special')) if RANK==0 else None
neuron.h('forall delete_section()')
neuron.load_mechanisms('mod/')
h.load_file('net_functions.hoc')
#===========================================================================
# Simulation, Analysis, and Plotting Controls
#===========================================================================
TESTING = False # i.e.g generate 1 cell/pop, with 0.1 s runtime
no_connectivity = False

stimulate = False # Add a stimulus
MDD = False #decrease PN GtonicApic and MN2PN weight by 40%
DRUG = False

rec_LFP = False #record LFP from center of layer
rec_DIPOLES = False #record population - wide dipoles

run_circuit_functions = False
#===========================================================================
# Params
#===========================================================================
dt = 0.025 #for both cell and network
tstart = 0.
tstop = 4500.
celsius = 34.
v_init = -80. #for both cell and network

###################### Load and organize Excel file ################################
circuit_params = {}

#Import Excel file
circuit_params = pd.read_excel('Circuit_param.xls', sheet_name = None, index_col = 0)

#Get cell names and import biophys
cell_names = [i for i in circuit_params['conn_probs'].axes[0]]
for name in cell_names:
    h.load_file('models/biophys_'+name+'.hoc')

circuit_params["syn_params"] = {'none':{'tau_r_AMPA': 0,'tau_d_AMPA': 0,'tau_r_NMDA': 0,
                                'tau_d_NMDA': 0, 'e': 0,'Dep': 0,'Fac': 0,'Use': 0,'u0':0,'gmax': 0}}
circuit_params["multi_syns"] = {'none':{'loc':0,'scale':0}}
# organizing dictionary for LFPY input
for pre in cell_names:
    for post in cell_names:
        if "PYR" in pre:
            circuit_params["syn_params"][pre+post] = {'tau_r_AMPA': 0.3, 'tau_d_AMPA': 3, 'tau_r_NMDA': 2,
                                                      'tau_d_NMDA': 65, 'e': 0, 'u0':0,
                                                      'Dep': circuit_params["Depression"].at[pre, post],
                                                      'Fac': circuit_params["Facilitation"].at[pre, post],
                                                      'Use': circuit_params["Use"].at[pre, post],
                                                      'gmax': circuit_params["syn_cond"].at[pre, post]}
        else:
            circuit_params["syn_params"][pre+post] = {'tau_r': 1, 'tau_d': 10, 'e': -80, 'u0':0,
                                                      'Dep': circuit_params["Depression"].at[pre, post],
                                                      'Fac': circuit_params["Facilitation"].at[pre, post],
                                                      'Use': circuit_params["Use"].at[pre, post],
                                                      'gmax': circuit_params["syn_cond"].at[pre, post]}
        circuit_params["multi_syns"][pre+post] = {'loc':int(circuit_params["n_cont"].at[pre, post]),'scale':0}


stimuli = []
for stimulus in circuit_params['STIM_PARAM'].axes[0]:
    stimuli.append({})
    for param_name in circuit_params['STIM_PARAM'].axes[1]:
        stimuli[-1][param_name] = circuit_params['STIM_PARAM'].at[stimulus, param_name]
    new_param = circuit_params["syn_params"][stimuli[-1]['syn_params']].copy()
    new_param['gmax'] = stimuli[-1]['gmax']
    stimuli[-1]['new_param'] = new_param
COMM.Barrier()
print('Importing, setting up MPI variables and loading necessary files took ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None
#################################################################################
if TESTING:
    OUTPUTPATH = 'Circuit_output_testing'
    for name in cell_names:
        circuit_params['SING_CELL_PARAM'].at['cell_num',name] = 1
    tstop = 100
    print('Running test...') if RANK ==0 else None

else:
    OUTPUTPATH = 'Circuit_output'
    print('Running full simulation...') if RANK==0 else None

COMM.Barrier()
##################################################################################
networkParams = {
        'dt' : dt,
        'tstart': tstart,
        'tstop' : tstop,
        'v_init' : v_init,
        'celsius' : celsius,
        'OUTPUTPATH' : OUTPUTPATH,
        'verbose': False}

#              L2/3   L4     L5
PYRmaxApics = [550   ,1550   ,1900]
uppers =      [-250  ,-1200 ,-1600]
lowers =      [-1200 ,-1580 ,-2300]

depths = []
rangedepths = []
minSynLocs = []
syn_pos = []
pop_args = {}

for i in range (3):
    depths.append((lowers[i]-uppers[i])/2-PYRmaxApics[i])
    rangedepths.append(abs(lowers[i]-uppers[i])/2)
    minSynLocs.append((lowers[i]-uppers[i])/2*3-PYRmaxApics[i])

    syn_pos.append({'section' : ['apic', 'dend'],
                    'fun' : [uniform_rv, halfnorm_rv],
                    'funargs' : [{'loc':minSynLocs[i], 'scale':abs(minSynLocs[i])},{'loc':minSynLocs[i], 'scale':abs(minSynLocs[i])}],
                    'funweights' : [1, 1.]})
    syn_pos.append({'section' : ['apic'],
                    'fun' : [uniform_rv],
                    'funargs' : [{'loc':minSynLocs[i], 'scale':abs(minSynLocs[i])}],
                    'funweights' : [1.]})
    syn_pos.append({'section' : ['dend'],
                    'fun' : [uniform_rv],
                    'funargs' : [{'loc':minSynLocs[i], 'scale':abs(minSynLocs[i])}],
                    'funweights' : [1.]})
    syn_pos.append({'section' : ['dend'],
                   'fun' : [halfnorm_rv],
                   'funargs' : [{'loc':minSynLocs[i], 'scale':abs(minSynLocs[i])}],
                   'funweights' : [1.]})
    names = ['HL2','HL4','HL5']
    pop_args[names[i]]={'radius':250,
                        'loc':depths[i],
                        'scale':rangedepths[i]*4,
                        'cap':rangedepths[i]}

# class RecExtElectrode parameters:
L23_size = abs(uppers[1] - lowers[1])
e1 = 5 #-725

LFPelectrodeParameters = dict(
        x=np.zeros(1),
        y=np.zeros(1),
        z=[e1],
        N=np.array([[0., 1., 0.] for _ in range(1)]),
        r=5.,
        n=50,
        sigma=0.3,
        method="soma_as_point")


#method Network.simulate() parameters

simargs = {'rec_imem': False,
           'rec_vmem': False,
           'rec_ipas': False,
           'rec_icap': False,
           'rec_isyn': False,
           'rec_vmemsyn': False,
           'rec_istim': False,
           'rec_current_dipole_moment':rec_DIPOLES,
           'rec_pop_contributions': False,
           'rec_variables': [],
           'to_memory': True,
           'to_file': False,
           'file_name':'OUTPUT.h5',
           'dotprodcoeffs': None}

#===========================================================================
# Functions
#===========================================================================
def generateSubPop(popsize,mname,popargs,Gou,Gtonic,GtonicApic):
    print('Initiating ' + mname + ' population...') if RANK==0 else None
    morphpath = 'morphologies/' + mname + '.swc'
    templatepath = 'models/NeuronTemplate.hoc'
    templatename = 'NeuronTemplate'
    pt3d = True
    cellParams = {
            'morphology': morphpath,
            'templatefile': templatepath,
            'templatename': templatename,
            'templateargs': morphpath,
            'v_init': v_init, #initial membrane potential, d=-65
            'passive': False,#initialize passive mechs, d=T, should be overwritten by biophys
            'dt': dt,
            'tstart': 0.,
            'tstop': tstop,#defaults to 100
            'nsegs_method': None,
            'pt3d': pt3d,#use pt3d-info of the cell geometries switch, d=F
            'delete_sections': False,
            'verbose': False}#verbose output switch, for some reason doens't work, figure out why}

    
    rotation = {'x':circuit_params['SING_CELL_PARAM'].at['rotate_x', mname],'y':circuit_params['SING_CELL_PARAM'].at['rotate_y', mname]}

    popParams = {
            'CWD': None,
            'CELLPATH': None,
            'Cell' : LFPy.NetworkCell, #play around with this, maybe put popargs into here
            'POP_SIZE': int(popsize),
            'name': mname,
            'cell_args' : {**cellParams},
            'pop_args' : popargs,
            'rotation_args' : rotation}

    network.create_population(**popParams)

    # Add biophys, OU processes, & tonic inhibition to cells
    for cellind in range(0,len(network.populations[mname].cells)):
        rseed = int(local_state.uniform()*SEED)
        biophys = 'h.biophys_' + mname + '(network.populations[\'' + mname + '\'].cells[' + str(cellind) + '].template)'
        exec(biophys)
        h.createArtificialSyn(rseed,network.populations[mname].cells[cellind].template,Gou)
        h.addTonicInhibition(network.populations[mname].cells[cellind].template,Gtonic,GtonicApic)

def addStimulus():
    cell_nums=[circuit_params['SING_CELL_PARAM'].at['cell_num',name] for name in cell_names]
    for stim in stimuli:
        stim_index = sum(cell_nums[:cell_names.index(stim['cell_name'])]) + stim['num_cells'] + stim['start_index']
        for gid, cell in zip(network.populations[stim['cell_name']].gids, network.populations[stim['cell_name']].cells):
            if gid < stim_index and gid >= sum(cell_nums[:cell_names.index(stim['cell_name'])]) + stim['start_index']:
                idx = cell.get_rand_idx_area_norm(section=stim['loc'], nidx=stim['loc_num'])
                for i in idx:
                    time_d=0
                    syn = Synapse(cell=cell, idx=i, syntype=stim['stim_type'], weight=1,**stim['new_param'])
                    while time_d <= 0:
                        time_d = np.random.uniform(low = stim['delay'], high = stim['delay']+stim['delay_range'])
                    syn.set_spike_times_w_netstim(noise=0, start=(stim['start_time']+time_d), number=stim['num_stim'], interval=stim['interval'], seed=GLOBALSEED)
#===========================================================================
# Sim
#===========================================================================
network = Network(**networkParams)

if MDD:
    # synaptic reduction
    for pre in cell_names:
        for post in cell_names:
            if 'SST' in pre:
                circuit_params["syn_params"][pre+post]["gmax"] = circuit_params["syn_cond"].at[pre, post]*0.6 # Synaptic reduction
    # tonic reduction
    for post in cell_names:
        if 'PYR' in post:
            circuit_params['SING_CELL_PARAM'].at['apic_tonic',post] = circuit_params['SING_CELL_PARAM'].at['apic_tonic',post]*0.6
            circuit_params['SING_CELL_PARAM'].at["drug_apic_tonic",post] = circuit_params['SING_CELL_PARAM'].at["drug_apic_tonic",post]*0.6
        else:
            sst = 0
            total = 0
            for pre in cell_names:
                if 'SST' in pre:
                    sst += circuit_params["syn_cond"].at[pre, post]*circuit_params["n_cont"].at[pre,post]*circuit_params["conn_probs"].at[pre, post]
                    total += circuit_params["syn_cond"].at[pre, post]*circuit_params["n_cont"].at[pre,post]*circuit_params["conn_probs"].at[pre, post]
                elif 'PV' in pre or 'VIP' in pre:
                    total += circuit_params["syn_cond"].at[pre, post]*circuit_params["n_cont"].at[pre, post]*circuit_params["conn_probs"].at[pre, post]
            circuit_params['SING_CELL_PARAM'].at['norm_tonic',post] -= circuit_params['SING_CELL_PARAM'].at['norm_tonic',post]*sst/total*0.4
            print(post, '_tonic reduced by: ', sst/total*100*0.4, '%') if RANK == 0 else None

# Generate Populations
tic = time.perf_counter()

for cell_name in cell_names:
    if DRUG:
        generateSubPop(circuit_params['SING_CELL_PARAM'].at['cell_num',cell_name],
                       cell_name,pop_args[cell_name[:3]],
                       circuit_params['SING_CELL_PARAM'].at['GOU',cell_name],
                       circuit_params['SING_CELL_PARAM'].at['drug_tonic',cell_name],
                       circuit_params['SING_CELL_PARAM'].at['drug_apic_tonic',cell_name])
    else:
        generateSubPop(circuit_params['SING_CELL_PARAM'].at['cell_num',cell_name],
                       cell_name,pop_args[cell_name[:3]],
                       circuit_params['SING_CELL_PARAM'].at['GOU',cell_name],
                       circuit_params['SING_CELL_PARAM'].at['norm_tonic',cell_name],               
                       circuit_params['SING_CELL_PARAM'].at['apic_tonic',cell_name])

COMM.Barrier()

print('Instantiating all populations took ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None

tic = time.perf_counter()

# Synaptic Connection Parameters
E_syn = neuron.h.ProbAMPANMDA
I_syn = neuron.h.ProbUDFsyn

for i, pre in enumerate(network.population_names):
    for j, post in enumerate(network.population_names):
        connectivity = network.get_connectivity_rand(
                        pre=pre,
                        post=post,
                        connprob=0 if no_connectivity else circuit_params["conn_probs"].at[pre, post])
        (conncount, syncount) = network.connect(
                        pre=pre, post=post,
                        connectivity=connectivity,
                        syntype=E_syn if "PYR" in pre else I_syn,
                        synparams=circuit_params["syn_params"][pre+post],
                        weightfun=local_state.normal,
                        weightargs={'loc':1, 'scale':0},
                        minweight=1,
                        delayfun=local_state.normal,
                        delayargs={'loc':0.5, 'scale':0},
                        mindelay=0.5,
                        multapsefun=local_state.normal,
                        multapseargs=circuit_params["multi_syns"][pre+post],

                        syn_pos_args=syn_pos[circuit_params["Syn_pos"].at[pre,post]])

print('Connecting populations took ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None

# Setup Extracellular Recording Device
COMM.Barrier()
if stimulate:
    addStimulus()
COMM.Barrier()
# Run Simulation

tic = time.perf_counter()

if rec_LFP and not rec_DIPOLES:
	  LFPelectrode = RecExtElectrode(**LFPelectrodeParameters)
	  print('Simulating, recording SPIKES and LFP ... ') if RANK==0 else None
	  SPIKES, OUTPUT, _ = network.simulate(electrode=LFPelectrode,**simargs)
elif rec_LFP and rec_DIPOLES:
	  print('Simulating, recording SPIKES, LFP, and DIPOLEMOMENTS ... ') if RANK==0 else None
	  LFPelectrode = RecExtElectrode(**LFPelectrodeParameters)
	  SPIKES, OUTPUT, DIPOLEMOMENT = network.simulate(electrode=LFPelectrode,**simargs)
elif not rec_LFP and rec_DIPOLES:
	  print('Simulating, recording SPIKES and DIPOLEMOMENTS ... ') if RANK==0 else None
	  SPIKES, _, DIPOLEMOMENT = network.simulate(**simargs)
elif not rec_LFP and not rec_DIPOLES:
	  print('Simulating, recording SPIKES ... ') if RANK==0 else None
	  SPIKES = network.simulate(**simargs)

print('Simulation took ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None

COMM.Barrier()
if RANK==0:
    tic = time.perf_counter()
    print('Saving simulation output ...')
    np.save(os.path.join(OUTPUTPATH,'SPIKES_Seed'+str(GLOBALSEED)+'.npy'), SPIKES)
    np.save(os.path.join(OUTPUTPATH,'OUTPUT_Seed'+str(GLOBALSEED)+'.npy'), OUTPUT) if rec_LFP else None
    np.save(os.path.join(OUTPUTPATH,'DIPOLEMOMENT_Seed'+str(GLOBALSEED)+'.npy'), DIPOLEMOMENT) if rec_DIPOLES else None
    print('Saving simulation took', str((time.perf_counter() - tic_0)/60)[:5], 'minutes')

#===========================================================================
# Plotting
#===========================================================================
if run_circuit_functions:
    tstart_plot = 2000
    tstop_plot = tstop
    print('Creating/saving plots...') if RANK==0 else None
    exec(open("circuit_functions.py").read())
#===============
#Final printouts
#===============
if TESTING:
    print('Test complete, switch TESTING to False for full simulation') if RANK==0 else None
elif not TESTING:
    print('Simulation complete') if RANK==0 else None
print('Script completed in ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None
