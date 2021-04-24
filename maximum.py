import argparse
import sys
import random

from collections import Counter

DEBUG=0
PLOT=0

def D(msg):
    if DEBUG: print(msg)

def flip(p):
    """returns True with a chance of p, False with a chance of 1-p

    :p: a float between 0 and 1
    :returns: a boolean

    """
    return random.random() < p

def to_dec(bits):
    """convert a 32-bit string to two decimals

    :bits: TODO
    :returns: TODO

    """
    return int(bits[:16], base=2), int(bits[16:], base=2)

def to_bin(n1, n2):
    return "{:016b}{:016b}".format(n1, n2)

# # test code
# n1 = 65535
# n2 = 0
# s = to_bin(n1, n2)
# assert(to_dec(s)[0] == n1)
# assert(to_dec(s)[1] == n1)


def random_bits(length):
    """generate a string of random bits (0 or 1) of length

    :length: TODO
    :returns: TODO

    """
    bits = ""
    for _ in range(length):
        bits += random.choice(['0','1'])
    return bits

def generate_population(size, bit_length=32):
    """returns a list of all population

    :size: the number of chromosomes in the population
    :bit_length: the length of chromosome
    :returns: a list of all population

    """
    return [ random_bits(bit_length) for _ in range(size) ]

def compute_fitness(bits):
    """compute the fitness of the string, specifically, the value of x^2 - y^2

    :bits: the bit string of length 32
    :returns: the fitness value

    """
    assert(len(bits) == 32)
    x = int(bits[:16], base=2)
    y = int(bits[16:], base=2)
    return x**2 - y**2

def compute_avg_fitness(lbits):
    """compute the average fitness of a list of strings

    :lbits: list of bits
    :returns: the fitness value

    """
    return sum([compute_fitness(bits) for bits in lbits]) / len(lbits)

def crossover(bitstr1, bitstr2, num=1):
    """crossover the two bitstrings at some random location

    :bitstr1: a 32-bit chromosome
    :bitstr2: a 32-bit chromosome
    :num: how many points to cross over
    :returns: TODO

    """
    def swap_at(bitstr1, bitstr2, at):
        """swap the latter part of the string from 'at'

        :bitstr1: TODO
        :bitstr2: TODO
        :at: TODO
        :returns: a tuple of the swapped strings

        """
        assert(at >= 0 and at < len(bitstr1))
        return (bitstr1[:at] + bitstr2[at:],
                bitstr2[:at] + bitstr1[at:])

    assert(len(bitstr1) == len(bitstr2))
    for i in range(num):
        location = random.randint(0, len(bitstr1)-1)
        bitstr1, bitstr2 = swap_at(bitstr1, bitstr2, location)
    return bitstr1, bitstr2

def mutate(bitstr, num=2):
    """mutate the bitstr

    :bitstr: TODO
    :num: number of bits to mutate
    :returns: TODO

    """
    for i in range(num):
        location = random.randint(0, len(bitstr)-1)
        bit = bitstr[location]
        bit = '1' if bit == '0' else '0'
        bitstr = bitstr[:location] + bit + bitstr[location+1:]
    return bitstr

def select(population, mutation_rate=0.5, crossover_rate=0.5, crossover_npoints=1, n_ary=2):
    """select and produce a fittest pair of offsprings from the old generation,
    Navarro-Barrientos, J-Emeterio. (2016). Re: What is 3-ary tournament selection?.
    Retrieved from: https://www.researchgate.net/post/What-is-3-ary-tournament-selection/56bdce767eddd3e5158b4581/citation/download.

    :population: TODO
    :mutation_rate: rate of mutating each of the produced offspring
    :crossover_rate: rate of crossing over the two fittest individuals
    :n_ary: the selection method.
    :returns: a list of the two offsprings

    """
    selection = random.choices(population, k=n_ary)
    D("selection 1: {}".format([(bits, compute_fitness(bits)) for bits in selection]))
    parent1 = max(selection, key=compute_fitness)
    selection = random.choices(population, k=n_ary)
    D("selection 2: {}".format([(bits, compute_fitness(bits)) for bits in selection]))
    parent2 = max(selection, key=compute_fitness)
    D("parent1: {}\nparent2: {}"
        .format(parent1, parent2))
    # # equivalent
    # parent1 = max(random.choices(population, k=n_ary), key=compute_fitness)
    # parent2 = max(random.choices(population, k=n_ary), key=compute_fitness)
    if (flip(crossover_rate)):
        D("Cross over...")
        parent1, parent2 = crossover(parent1, parent2, crossover_npoints)
        D("parent1: {}\nparent2: {}".format(parent1, parent2))
    if (flip(mutation_rate)):
        parent1 = mutate(parent1)
        D("mutate 1...")
        D("parent1: {}".format(parent1))

    if (flip(mutation_rate)):
        parent2 = mutate(parent2)
        D("mutate 2...")
        D("parent2: {}".format(parent2))
    return [ parent1, parent2 ]

def has_converged(population, p):
    """determines whether a population has converged

    :population: TODO
    :p: TODO
    :returns: TODO

    """
    c = Counter(population)
    value, count = c.most_common(1)[0]
    ret = count/len(population) > p
    if ret:
        print("Population converged with {:.2f}% population with the value {}"
              .format(count/len(population)*100, value))
    else:
        print("{:.2f}% population with the value {}"
              .format(count/len(population)*100, value))

    return ret

def main():
    parser = argparse.ArgumentParser(description="Perform a genetic algorithm to maximize the value of x^2 - y^2")
    parser.add_argument("--show-plot", dest="show_plot", action="store_true", help="Show a plot of fitness over each generation")
    parser.add_argument("--no-show-plot", dest="show_plot", action="store_false", help="Show a plot of fitness over each generation")
    parser.set_defaults(show_plot=True)
    parser.add_argument("--max-generation", type=int, default=50000, help="The max number of generations")
    parser.add_argument("--convergence-level", type=float, default=0.95,
                        help="The percentage of common value in population that defines convergence of the algorithm")
    parser.add_argument("--population-size", type=int, default=10000, help="The population size")
    parser.add_argument("--mutation-rate", type=float, default=0.05, help="The mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.9, help="The crossover rate")
    parser.add_argument("--crossover-npoints", type=int, default=2, help="How many points to use in crossover (e.g. 1, 2)")
    parser.add_argument("--n-ary", type=int, default=5, help="N-ary selection number (e.g. 2, 3)")

    args = parser.parse_args()

    global PLOT
    PLOT = args.show_plot

    print(vars(args))

    population = generate_population(args.population_size)
    new_population = []
    if PLOT:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as e:
            print("Turning off plotting feature.")
            PLOT = False
    if PLOT:
        fig, axes = plt.subplots()
        axes.set_title("fitness of generations")
        axes.set_xlabel("generation")
        axes.set_ylabel("fitness")
    for i in range(args.max_generation):
        avg_fitness = compute_avg_fitness(population)
        if PLOT:
            axes.plot(i, avg_fitness, "bo")
        print("***GENERATION {}".format(i))
        print("avg_fitness: {}".format(avg_fitness))
        best = max(population, key=compute_fitness)
        print("best: {}".format(to_dec(best)))
        print("fitness: {}".format(compute_fitness(best)))
        while len(new_population) < args.population_size:
            new_population += select(
                population,
                args.mutation_rate,
                args.crossover_rate,
                args.crossover_npoints,
                args.n_ary)
        if has_converged(new_population, args.convergence_level):
            return
        population = new_population
        new_population = []
        if PLOT:
            plt.pause(0.05)

main()
