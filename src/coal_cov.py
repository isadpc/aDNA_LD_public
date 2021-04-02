"""Classes to simulate two-locus tree statistics."""

import numpy as np
import msprime as msp


class TwoLocusSimulation:
    """Base class representing a two-locus simulation."""

    def __init__(self, Ne=1e4, rec_rate=1e-4, n=2, loci=2, reps=10):
        """Initialize the two-locus simulation class."""
        self.Ne = Ne
        self.rec_rate = rec_rate
        self.n = n
        self.loci = loci
        self.reps = reps
        self.treeseq = None
        self.samples = [msp.Sample(population=0, time=0) for i in range(n)]

    def _simulate(self, **kwargs):
        """Simulate a two-locus genealogy.

        NOTE: will be re-written in each subsequent simulation class per scenario)

        """
        recomb_map = msp.RecombinationMap.uniform_map(
            self.loci, self.rec_rate, num_loci=self.loci
        )
        self.treeseq = msp.simulate(
            sample_size=self.n,
            Ne=self.Ne,
            recombination_map=recomb_map,
            num_replicates=self.reps,
            **kwargs
        )

    def _two_locus_tmrca(self):
        """Calculate the tree height at each locus."""
        assert self.treeseq is not None
        pair_tmrca = np.zeros(shape=(self.reps, 2))
        i = 0
        for ts in self.treeseq:
            # TODO : there might be a faster way to do this operation as well?
            cur_times = [t.time(t.root) for t in ts.trees()]
            cur_times = np.tile(cur_times, 2)
            pair_tmrca[i] = cur_times[:2]
            i += 1

        self.pair_tmrca = pair_tmrca
        self.pair_tmrca = self.pair_tmrca / (2.0 * self.Ne)
        self.treeseq = None

    def _two_locus_branch_length(self, scale=False):
        """Calculate the total tree length at each locus."""
        assert self.treeseq is not None
        pair_branch_length = np.zeros(shape=(self.reps, 2))
        i = 0
        for ts in self.treeseq:
            # NOTE : using the tskit approach here because it is coded in c
            cur_bl = ts.segregating_sites(mode="branch", windows="trees")
            cur_bl = np.tile(cur_bl, 2)
            pair_branch_length[i] = cur_bl[:2]
            i += 1

        self.pair_branch_length = pair_branch_length
        if scale:
            self.pair_branch_length = self.pair_branch_length / (2.0 * self.Ne)
        # reset the iterator
        self.treeseq = None


class TwoLocusSerialCoalescent(TwoLocusSimulation):
    """Class for two locus simulations with serial samples."""

    def __init__(self, ta, Ne=1e4, rec_rate=1e-4, na=1, n0=1, loci=2, reps=10):
        """Initialize the class.

        Arguments:
        -----------
          ta - float
            sampling time for ancient individuals

          Ne - float
            effective population size
          rec_rate - float
            recombination rate between the two loci

          na - int
            number of ancient haplotypes to sample

          n0 - int
            number of modern haplotypes to sample

        """
        self.Ne = Ne
        self.ta = ta
        self.loci = loci
        self.rec_rate = rec_rate
        self.samples1 = [msp.Sample(population=0, time=0) for i in range(n0)]
        self.samples2 = [msp.Sample(population=0, time=ta) for i in range(na)]
        self.samples = self.samples1 + self.samples2
        self.reps = reps
        self.treeseq = None

    def _simulate(self, **kwargs):
        recomb_map = msp.RecombinationMap.uniform_map(
            self.loci, self.rec_rate/self.loci, num_loci=self.loci
        )
        ts = msp.simulate(
            Ne=self.Ne,
            samples=self.samples,
            recombination_map=recomb_map,
            num_replicates=self.reps, 
            **kwargs
        )
        return(ts)


class TwoLocusSerialDivergence(TwoLocusSimulation):
    """Simulation of Two-Locus model with divergence and serial sampling."""

    def __init__(
        self, ta, Ne=1e4, t_div=0.0, rec_rate=1e-4, na=1, n0=1, eps=1e-6, reps=100
    ):
        """Initialize the model with serial sampling.

        Arguments:
        -----------
          ta - float
            sampling time for ancient individuals

          Ne - float
            effective population size

          t_div - float
            time of split between population A and B

          rec_rate - float
            recombination rate between the two loci

          na - int
            number of ancient haplotypes to sample

          n0 - int
            number of modern haplotypes to sample

        """
        self.Ne = Ne
        self.ta = ta
        self.loci = 2
        self.tmrcas = []
        self.rec_rate = rec_rate
        self.reps = reps
        self.t_div = t_div
        self.samples1 = [msp.Sample(population=0, time=0) for i in range(n0)]
        self.samples2 = [msp.Sample(population=1, time=ta) for i in range(na)]
        self.samples = self.samples1 + self.samples2
        self.pop_config = [msp.PopulationConfiguration(), msp.PopulationConfiguration()]
        self.demography = [msp.MassMigration(time=(ta + t_div + eps), source=1, dest=0)]
        self.treeseq = None

    def _simulate(self, **kwargs):
        # Define a recombination map with two loci
        recomb_map = msp.RecombinationMap.uniform_map(
            self.loci, self.rec_rate, num_loci=self.loci
        )
        self.treeseq = msp.simulate(
            Ne=self.Ne,
            samples=self.samples,
            population_configurations=self.pop_config,
            demographic_events=self.demography,
            recombination_map=recomb_map,
            num_replicates=self.reps,
        )


class TwoLocusSerialTennessen(TwoLocusSimulation):
    """Simulate a Two-Locus system for two European individuals under the Tennessen et al model."""

    def __init__(self, ta, n0=1, na=1, Ne=1e4, rec_rate=1e-4, loci=2, reps=100):
        """Initialize the model."""
        generation_time = 25
        T_AF = 148e3 / generation_time
        T_OOA = 51e3 / generation_time
        T_EU0 = 23e3 / generation_time
        T_EG = 5115 / generation_time

        # Growth rates
        r_EU0 = 0.00307
        r_EU = 0.0195
        r_AF = 0.0166

        # population sizes
        N_A = 7310
        N_AF1 = 14474
        N_B = 1861
        N_EU0 = 1032
        N_EU1 = N_EU0 / np.exp(-r_EU0 * (T_EU0 - T_EG))

        # migration rates
        m_AF_B = 15e-5
        m_AF_EU = 2.5e-5

        # present Ne
        N_EU = N_EU1 / np.exp(-r_EU * T_EG)
        N_AF = N_AF1 / np.exp(-r_AF * T_EG)

        population_configurations = [
            msp.PopulationConfiguration(initial_size=N_AF, growth_rate=r_AF),
            msp.PopulationConfiguration(initial_size=N_EU, growth_rate=r_EU),
        ]
        migration_matrix = [[0, m_AF_EU], [m_AF_EU, 0]]
        demographic_events = [
            msp.MigrationRateChange(time=T_EG, rate=m_AF_EU, matrix_index=(0, 1)),
            msp.MigrationRateChange(time=T_EG, rate=m_AF_EU, matrix_index=(1, 0)),
            msp.PopulationParametersChange(
                time=T_EG, growth_rate=r_EU0, initial_size=N_EU1, population_id=1
            ),
            msp.PopulationParametersChange(
                time=T_EG, growth_rate=0, initial_size=N_AF1, population_id=0
            ),
            msp.MigrationRateChange(time=T_EU0, rate=m_AF_B, matrix_index=(0, 1)),
            msp.MigrationRateChange(time=T_EU0, rate=m_AF_B, matrix_index=(1, 0)),
            msp.PopulationParametersChange(
                time=T_EU0, initial_size=N_B, growth_rate=0, population_id=1
            ),
            msp.MassMigration(time=T_OOA, source=1, destination=0, proportion=1.0),
            msp.PopulationParametersChange(
                time=T_AF, initial_size=N_A, population_id=0
            ),
        ]

        self.pop_config = population_configurations
        self.migration_matrix = migration_matrix
        self.demography = demographic_events
        self.rec_rate = rec_rate
        self.loci = loci
        self.samples1 = [msp.Sample(population=1, time=0) for i in range(n0)]
        self.samples2 = [msp.Sample(population=1, time=ta) for i in range(na)]
        self.samples = self.samples1 + self.samples2
        self.reps = reps
        self.Ne = Ne
        self.treeseq = None

    def _simulate(self, **kwargs):
        # Define a recombination map with two loci
        recomb_map = msp.RecombinationMap.uniform_map(
            self.loci, self.rec_rate, num_loci=self.loci
        )
        self.treeseq = msp.simulate(
            Ne=self.Ne,
            samples=self.samples,
            population_configurations=self.pop_config,
            demographic_events=self.demography,
            recombination_map=recomb_map,
            num_replicates=self.reps,
        )

    def _demography_debug(self):
        """Demography debugging."""
        dd = msp.DemographyDebugger(
            population_configurations=self.pop_config,
            migration_matrix=self.migration_matrix,
            demographic_events=self.demography,
        )
        # print out the debugging history
        dd.print_history()


class TwoLocusSerialIBDNeUK10K(TwoLocusSimulation):
    """Two-locus simulation under UK10K demography from Browning and Browning."""

    def __init__(self, ta, demo_file=None, loci=2, na=1, n0=1, rec_rate=1e-4, reps=100):
        """Initialize the model."""
        super().__init__()
        self.Ne = 1e4
        assert demo_file is not None
        self.demo_file = demo_file
        self.loci = loci
        self.rec_rate = rec_rate
        self.reps = reps
        self.samples1 = [msp.Sample(population=0, time=0) for i in range(n0)]
        self.samples2 = [msp.Sample(population=0, time=ta) for i in range(na)]
        self.samples = self.samples1 + self.samples2

    def _set_demography(self):
        demography = []
        N0 = None
        line_num = 0
        t = 0
        for line in open(self.demo_file, "r+"):
            spltln = line.split()
            Nt = int(spltln[1])
            if spltln[2] != "inf":
                deltat = int(spltln[2])
                if line_num == 0:
                    N0 = Nt
                line_num += 1
                t += deltat
                demography.append(
                    msp.PopulationParametersChange(time=t, initial_size=Nt)
                )

        self.pop_config = [msp.PopulationConfiguration(initial_size=N0)]
        self.demography = demography

    def _simulate(self, **kwargs):
        recomb_map = msp.RecombinationMap.uniform_map(
            self.loci, self.rec_rate, num_loci=self.loci
        )
        self.treeseq = msp.simulate(
            Ne=self.Ne,
            samples=self.samples,
            demographic_events=self.demography,
            population_configurations=self.pop_config,
            recombination_map=recomb_map,
            num_replicates=self.reps,
        )

    def _demography_debug(self):
        """Demography debugging."""
        dd = msp.DemographyDebugger(
            population_configurations=self.pop_config,
            demographic_events=self.demography,
        )
        # print out the debugging history
        dd.print_history()


class TwoLocusSerialGrowth(TwoLocusSimulation):
    """Simulate a Two-Locus system under population growth."""

    def __init__(
        self, n=1, Ne=1e4, ta=100, na=1, rec_rate=1e-4, loci=2, reps=100, T=100, r=0.1
    ):
        """Initializ the simulation."""
        Ne0 = Ne / np.exp(-r * T)

        population_configurations = [
            msp.PopulationConfiguration(initial_size=Ne0, growth_rate=r)
        ]

        self.pop_config = population_configurations
        self.rec_rate = rec_rate
        self.loci = loci
        self.samples1 = [msp.Sample(population=0, time=0) for i in range(n)]
        self.samples2 = [msp.Sample(population=0, time=ta) for j in range(na)]
        self.samples = self.samples1 + self.samples2
        self.reps = reps
        self.Ne = Ne
        self.treeseq = None

    def _simulate(self, **kwargs):
        # Define a recombination map with two loci
        recomb_map = msp.RecombinationMap.uniform_map(
            self.loci, self.rec_rate, num_loci=self.loci
        )
        self.treeseq = msp.simulate(
            Ne=self.Ne,
            samples=self.samples,
            population_configurations=self.pop_config,
            recombination_map=recomb_map,
            num_replicates=self.reps,
        )


class TwoLocusSerialBottleneck(TwoLocusSimulation):
    """Simulate a Two-Locus system with a bottleneck."""

    def __init__(
        self,
        Ne=1e4,
        ta=100,
        n0=1,
        na=1,
        Nbot=1e3,
        Tstart=100,
        Tend=200,
        rec_rate=1e-4,
        loci=2,
        reps=100,
    ):
        """Initialize the model."""
        assert Tstart < Tend
        population_configurations = [
            msp.PopulationConfiguration(initial_size=Ne, growth_rate=0)
        ]

        demographic_events = [
            msp.PopulationParametersChange(
                time=Tstart, initial_size=Nbot, population_id=0
            ),
            msp.PopulationParametersChange(time=Tend, initial_size=Ne, population_id=0),
        ]

        self.pop_config = population_configurations
        self.demography = demographic_events
        self.rec_rate = rec_rate
        self.loci = loci
        self.samples1 = [msp.Sample(population=0, time=0) for i in range(n0)]
        self.samples2 = [msp.Sample(population=0, time=ta) for i in range(na)]
        self.samples = self.samples1 + self.samples2
        self.reps = reps
        self.Ne = Ne
        self.treeseq = None

    def _simulate(self, **kwargs):
        # Define a recombination map with two loci
        recomb_map = msp.RecombinationMap.uniform_map(
            self.loci, self.rec_rate, num_loci=self.loci
        )
        self.treeseq = msp.simulate(
            Ne=self.Ne,
            samples=self.samples,
            population_configurations=self.pop_config,
            demographic_events=self.demography,
            recombination_map=recomb_map,
            num_replicates=self.reps,
        )

    def _demography_debug(self):
        """Demography debugging."""
        dd = msp.DemographyDebugger(
            population_configurations=self.pop_config,
            demographic_events=self.demography,
        )
        # print out the debugging history
        dd.print_history()


class TwoLocusTheoryConstant:
    """Theoretical two-locus properties in a model of population-continuity."""
    def _p100_to_011(rho,ta):
        return(r * (1.0 - np.exp(-t * (r / 2 + 1))) / (r + 2.))

    def _p011_to_100(rho,ta):
        return(2. * (1.0 - np.exp(-t * (r / 2 + 1))) / (r + 2.))

    def _eTATB(rho, ta):
        u200 = lambda rho: (rho ** 2 + 14 * rho + 36) / (
            rho ** 2 + 13 * rho + 18
        )  # noqa
        u111 = lambda rho: (rho ** 2 + 13 * rho + 24) / (
            rho ** 2 + 13 * rho + 18
        )  # noqa
        # Calculate the probability of uncoupling of the ancient haplotype
        p111 = lambda r, t: r * (1.0 - np.exp(-t * (r / 2 + 1))) / (r + 2)  # noqa
        return p111(rho, ta) * u111(rho) + (1 - p111(rho, ta)) * u200(rho)

    def _eTATB_appx(rho, ta):
        u200 = lambda rho: (rho ** 2 + 14 * rho + 36) / (
            rho ** 2 + 13 * rho + 18
        )  # noqa
        u111 = lambda rho: (rho ** 2 + 13 * rho + 24) / (
            rho ** 2 + 13 * rho + 18
        )  # noqa
        # Calculate the probability of uncoupling of the ancient haplotype
        p111_appx = lambda r, t: (t * r) / 2  # noqa
        return p111_appx(rho, ta) * u111(rho) + (1 - p111_appx(rho, ta)) * u200(rho)

    def _corrLALB(rho, ta):
        return TwoLocusTheoryConstant._eTATB(rho, ta) - 1.0

    def _corrLALB_appx(rho, ta):
        return TwoLocusTheoryConstant._eTATB_appx(rho, ta) - 1.0

    def _covLALB(rho, ta):
        return 4 * (TwoLocusTheoryConstant._corrLALB(rho, ta))
    
    def _covSASB(rho, ta, theta=1.0):
        return (theta**2)/4. *(TwoLocusTheoryConstant._covLALB(rho, ta))
    
    def _corrSASB(rho, ta, theta=1.0):
        """Correlation in segregating sites."""
        corrSASB = (
            1.0
            / (1.0 + (2.0 + ta) / (2 * theta))
            * TwoLocusTheoryConstant._corrLALB(rho, ta)
        )
        return corrSASB


class TwoLocusTheoryDivergence:
    """Class for theoretical moments of two-locus properties in a model of simple divergence."""

    def _eTATB(rho, ta, tdiv):
        """Estimate of the joint moment of the underlying."""
        p111 = lambda r, t: r * (1.0 - np.exp(-t * (r / 2.0 + 1.0))) / (r + 2.0)  # noqa
        p200 = lambda r, t: 1.0 - p111(r, t)  # noqa

        # theoretical properties
        u200 = lambda rho: (rho ** 2 + 14 * rho + 36) / (
            rho ** 2 + 13 * rho + 18
        )  # noqa
        u111 = lambda rho: (rho ** 2 + 13 * rho + 24) / (
            rho ** 2 + 13 * rho + 18
        )  # noqa
        u022 = lambda rho: (rho ** 2 + 13 * rho + 22) / (
            rho ** 2 + 13 * rho + 18
        )  # noqa

        a1 = p200(rho, ta + tdiv) * p200(rho, tdiv)
        a2 = p111(rho, ta + tdiv) * p111(rho, tdiv)
        a3 = p200(rho, ta + tdiv) * p111(rho, tdiv)
        a4 = p200(rho, tdiv) * p111(rho, ta + tdiv)
        # Joint distribution of tree heights
        eTATB = a1 * u200(rho)
        eTATB = eTATB + a2 * u022(rho)
        eTATB = eTATB + (a3 + a4) * u111(rho)
        return eTATB

    def _corrLALB(rho, ta, tdiv):
        """Correlation in total branch length."""
        return TwoLocusTheoryDivergence._eTATB(rho, ta, tdiv) - 1.0

    def _corrSASB(rho, ta, tdiv, theta=1.0):
        """Correlation in segregating sites."""
        corrSASB = (
            1.0
            / (1.0 + (2 + ta) / (2 * theta))
            * TwoLocusTheoryDivergence._corrLALB(rho, ta, tdiv)
        )
        return corrSASB
