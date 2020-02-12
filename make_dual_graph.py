import geopandas as gpd
import pickle
from gerrychain import Graph, Partition

df = gpd.read_file("GA_precincts16/GA_precincts16.shp")
df = df.astype({"SEN16D": "int64", "SEN16R": "int64",
                "PRES16D": "int64", "PRES16R": "int64"})
df.to_file("GA_precincts16/GA_precincts16.shp")

graph = Graph.from_file("GA_precincts16/GA_precincts16.shp")

with open("GA_precinct_graph.p", "wb") as f_out:
    pickle.dump(graph, f_out)