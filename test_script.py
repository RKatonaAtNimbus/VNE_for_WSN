"""
Takes as input a pickle file.
Processes all the input vectors.
Generates a solution vector file and pickles it.
"""
import pickle
import test_vne
import sys

solution_path = "solution_vector_{size}_{vnrs}.pickle"

print "Loading IV",
input_vector = pickle.load(open(sys.argv[1]))
print "done"

solution_vector = []

nwk_size = input_vector[0]['nwksize']
numvns = input_vector[0]['numvn']

for iv in input_vector:
    print "\033[91m Solving", iv['nwksize'], iv['numvn'], iv['iteration'], "\033[0m"
    # Objective 1: Maximize acceptance ratio
    solution = test_vne.algorithm(iv)
    if solution is None:
        print "Input generation error for", iv['nwksize'], iv['numvn'], iv['iteration']
        continue
    accepted = len(solution['milp_sol']['mapping'])
    # Objective 2: Minimize cost
    solution = test_vne.find_best_cost_fixed_acceptance(iv, accepted)
    solution_vector.append(solution)

print "Done. Pickling..."
pickle.dump(solution_vector,
            open(solution_path.format(size=nwk_size, vnrs=numvns), 'w'))

print "All done. Bye!"
