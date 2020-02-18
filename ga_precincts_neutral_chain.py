import argparse
from gerrychain import Graph, GeographicPartition, Partition, Election, accept
from gerrychain.updaters import Tally, cut_edges
from gerrychain import MarkovChain
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from gerrychain import constraints
import geopandas as gpd
import numpy as np
from functools import partial
from gerrychain.tree import recursive_tree_part
import pickle
import random

## Set up argument parser

parser = argparse.ArgumentParser(description="Neutral ensemble for GA", 
                                 prog="ga_precincts_neutral_chain.py")
parser.add_argument("map", metavar="map", type=str,
                    choices=["congress", #"congress_2020",
                             "state_house", "state_senate"],
                    help="the map to redistrict")
parser.add_argument("n", metavar="iterations", type=int,
                    help="the number of plans to sample")
args = parser.parse_args()

num_districts_in_map = {"congress" : 14,
                        "state_senate" : 56,
                        "state_house" : 180}

epsilons = {"congress" : 0.01,
            "congress_2020" : 0.01,
            "state_senate" : 0.02,
            "state_house" : 0.05} 

POP_COL = "TOTPOP"
NUM_DISTRICTS = num_districts_in_map[args.map]
ITERS = args.n
EPS = epsilons[args.map]
ELECTS = ["PRES16", "SEN16"]

## Pull in graph and set up updaters

print("Reading in Data/Graph")

df = gpd.read_file("GA_precincts16/GA_precincts16.shp")
with open("GA_precinct_graph.p", "rb") as f_in:
    graph = pickle.load(f_in)

elections = [Election("SEN16", {"Dem": "SEN16D", "Rep": "SEN16R"}),
             Election("PRES16", {"Dem": "PRES16D", "Rep": "PRES16R"})]


ga_updaters = {"population" : Tally(POP_COL, alias="population"),
               "cut_edges": cut_edges,
               "VAP": Tally("VAP"),
               "WVAP": Tally("WVAP"),
               "HVAP": Tally("HVAP"),
               "BVAP": Tally("BVAP"),
               "HVAP_perc": lambda p: {k: (v / p["VAP"][k]) for k, v in p["HVAP"].items()},
               "WVAP_perc": lambda p: {k: (v / p["VAP"][k]) for k, v in p["WVAP"].items()},
               "BVAP_perc": lambda p: {k: (v / p["VAP"][k]) for k, v in p["BVAP"].items()},
               "BHVAP_perc": lambda p: {k: ((p["HVAP"][k] + p["BVAP"][k]) / v) for k, v in p["VAP"].items()},}

election_updaters = {election.name: election for election in elections}
ga_updaters.update(election_updaters)

## Create seed plans and Set up Markov chain

print("Creating seed plan")

total_pop = sum(df[POP_COL])
ideal_pop = total_pop / NUM_DISTRICTS

if args.map != "state_house":
    cddict = recursive_tree_part(graph=graph, parts=range(NUM_DISTRICTS), 
                                 pop_target=ideal_pop, pop_col=POP_COL, epsilon=EPS)
else:
    with open("GA_house_seed_part_0.05.p", "rb") as f:
        cddict = pickle.load(f)

init_partition = Partition(graph, assignment=cddict, updaters=ga_updaters)


## Setup chain

proposal = partial(recom, pop_col=POP_COL, pop_target=ideal_pop, epsilon=EPS, 
                   node_repeats=1)

compactness_bound = constraints.UpperBound(lambda p: len(p["cut_edges"]), 
                                           2*len(init_partition["cut_edges"]))

chain = MarkovChain(
        proposal,
        constraints=[
            constraints.within_percent_of_ideal_population(init_partition, EPS),
            compactness_bound],
        accept=accept.always_accept,
        initial_state=init_partition,
        total_steps=ITERS)


## Run chain

print("Starting Markov Chain")

def init_chain_results(elections):
    data = {"cutedges": np.zeros(ITERS)}
    parts = {"samples": [], "compact": []}

    data["HVAP"] = np.zeros((ITERS, NUM_DISTRICTS))
    data["BVAP"] = np.zeros((ITERS, NUM_DISTRICTS))
    data["HVAP_perc"] = np.zeros((ITERS, NUM_DISTRICTS))
    data["BVAP_perc"] = np.zeros((ITERS, NUM_DISTRICTS))
    data["BHVAP_perc"] = np.zeros((ITERS, NUM_DISTRICTS))
    data["WVAP"] = np.zeros((ITERS, NUM_DISTRICTS))
    data["WVAP_perc"] = np.zeros((ITERS, NUM_DISTRICTS))

    for election in elections:
        name = election.lower()
        data["seats_{}".format(name)] = np.zeros(ITERS)
        data["results_{}".format(name)] = np.zeros((ITERS, NUM_DISTRICTS))
        data["efficiency_gap_{}".format(name)] = np.zeros(ITERS)
        data["mean_median_{}".format(name)] = np.zeros(ITERS)
        data["partisan_gini_{}".format(name)] = np.zeros(ITERS)

    return data, parts

def tract_chain_results(data, elections, part, i):
    data["cutedges"][i] = len(part["cut_edges"])

    data["HVAP"][i] = sorted(part["HVAP"].values())
    data["BVAP"][i] = sorted(part["BVAP"].values())
    data["HVAP_perc"][i] = sorted(part["HVAP_perc"].values())
    data["BVAP_perc"][i] = sorted(part["BVAP_perc"].values())
    data["BHVAP_perc"][i] = sorted(part["BHVAP_perc"].values())
    data["WVAP"][i] = sorted(part["WVAP"].values())
    data["WVAP_perc"][i] = sorted(part["WVAP_perc"].values())

    for election in elections:
        name = election.lower()
        data["results_{}".format(name)][i] = sorted(part[election].percents("Dem"))
        data["seats_{}".format(name)][i] = part[election].seats("Dem")
        data["efficiency_gap_{}".format(name)][i] = part[election].efficiency_gap()
        data["mean_median_{}".format(name)][i] = part[election].mean_median()
        data["partisan_gini_{}".format(name)][i] = part[election].partisan_gini()


def update_saved_parts(parts, part, elections, i):
    if i % (ITERS / 10) == 99: parts["samples"].append(part.assignment)


chain_results, parts = init_chain_results(ELECTS)

for i, part in enumerate(chain):
    chain_results["cutedges"][i] = len(part["cut_edges"])
    tract_chain_results(chain_results, ELECTS, part, i)
    update_saved_parts(parts, part, ELECTS, i)

    if i % 1000 == 0:
        print("*", end="", flush=True)
print()

## Save results

print("Saving results")

output = "/cluster/tufts/mggg/jmatth03/Georgia/GA_{}_{}.p".format(args.map, ITERS)
output_parts = "/cluster/tufts/mggg/jmatth03/Georgia/GA_{}_{}_parts.p".format(args.map, ITERS)

with open(output, "wb") as f_out:
    pickle.dump(chain_results, f_out)

with open(output_parts, "wb") as f_out:
    pickle.dump(parts, f_out)
