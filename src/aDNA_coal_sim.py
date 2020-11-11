"""Helpful functions for simulating serial samples."""

import numpy as np
import msprime as msp


class Simulation(object):
    """Parameters for a simulation object."""

    def __init__(self):
        """Create a simulation object."""
        # Defining effective population sizes
        self.Ne = None

        # Defining samples in this case
        self.samples = None

        # Define demographic events to sample across
        self.demography = None

        # Define population configuration
        self.pop_config = None

    def _simulate(self, **kwargs):
        """Conduct a simulation using msprime and the parameters we have."""
        # Generate a tree sequence
        tree_seq = msp.simulate(
            Ne=self.Ne,
            samples=self.samples,
            demographic_events=self.demography,
            population_configurations=self.pop_config,
            **kwargs
        )
        return tree_seq

    def _demography_debug(self):
        """Demography debugging."""
        dd = msp.DemographyDebugger(
            population_configurations=self.pop_config,
            demographic_events=self.demography,
        )
        dd.print_history()

    def _pos_r2(ts):
        """Obtain vectors of position differences and r^2 per pair of sites.

        Arguments
        ---------
        ts : msprime.TreeSequence
            tree sequence object

        Returns
        -------
        pos_diff : np.array
            position difference for pairs of snps

        r2 : np.array
            r^2 as computed between the different sites

        """
        ld_calc = msp.LdCalculator(ts)
        r2_est = ld_calc.r2_matrix()
        # Computing positions and indices
        pos = np.array([s.position for s in ts.sites()], dtype=np.float32)
        pos_diff_mat = np.zeros(shape=(pos.shape[0], pos.shape[0]), dtype=np.float32)
        for i in np.arange(len(pos)):
            for j in np.arange(i):
                # Calculating the absolute difference in position
                pos_diff_mat[i, j] = np.abs(pos[i] - pos[j])

        # Extract entries that matter (and are matched)
        r2 = r2_est[pos_diff_mat > 0]
        pos_diff = pos_diff_mat[pos_diff_mat > 0]

        # Set undefined values to be 1 (due to non-segregating issues...)
        r2[np.isnan(r2)] = 1.0
        return (pos_diff, r2)

    def filt_time_points(ts, ti, **kwargs):
        """Filter to samples at a particular point in time.

        Arguments
        ---------
        ts : msprime.TreeSequence
            tree sequence object

        ti : float
            time-point for sampling

        Returns
        -------
        ts : msprime.TreeSequence
            modified tree-sequence structure with ascertainment

        """
        sample_list = []
        t = ts.first()
        for n in t.samples():
            if t.time(n) == ti:
                sample_list.append(n)
        return ts.simplify(samples=sample_list, **kwargs)

    def _ploidy_genotype_matrix(ts, ploidy=2):
        """Generate diploid genotype matrices from TreeSequences."""
        geno = ts.genotype_matrix().T
        n, p = geno.shape
        assert n % ploidy == 0
        true_n = int(n / 2)
        true_geno = np.zeros(shape=(true_n, p))
        for i in range(true_n):
            x = ploidy * i
            y = ploidy * i + ploidy
            true_geno[i] = np.sum(geno[x:y, :], axis=0)
        return true_geno


class SerialConstant(Simulation):
    """Simulate serial haplotypes under constant pop size."""

    def __init__(self, Ne, mod_n, t_anc, n_anc):
        """Class that defines serially sampled haplotypes.

        NOTE : this is for a constant sized population

        Arguments
        ---------
        Ne : np.float32
          Effective population size

        mod_n : np.int32
          modern sample size

        t_anc : list (np.float32)
          list of floats that dictate sampling times

        n_anc : list (np.int32)
          list of sample sizes for individuals

        """
        super().__init__()

        # Define effective population size
        self.Ne = Ne

        # Setup samples
        assert len(t_anc) == len(n_anc)  # need to make sure these match up
        samples = [msp.Sample(population=0, time=0) for i in range(mod_n)]
        for (t, n) in zip(t_anc, n_anc):
            for i in range(n):
                # Append samples in the same
                samples.append(msp.Sample(0, t))

        # Set samples as the parameters for the object
        self.samples = samples


class SerialBottleneck(Simulation):
    """Simulate a bottleneck with ancient samples."""

    def __init__(
        self,
        Ne,
        mod_n,
        t_anc,
        n_anc,
        bottle_start=200,
        bottle_duration=300,
        bottle_mag=0.1,
    ):
        """Class defining a single population model with a bottleneck.

        Params
        ------
        * t_bottleneck : timing of the bottleneck
        * t_bottle : duration of the bottleneck
        * mag_bottle : magnitude of the bottleneck

        """
        super().__init__()

        # Define effective population size
        self.Ne = Ne

        # Setup samples here
        assert len(t_anc) == len(n_anc)

        samples = [msp.Sample(population=0, time=0) for i in range(mod_n)]
        for (t, n) in zip(t_anc, n_anc):
            for i in range(n):
                # Append ancient samples in the other population
                samples.append(msp.Sample(population=0, time=t))

        self.samples = samples

        # Define the bottlneck as a demographic event here
        self.demography = [
            msp.PopulationParametersChange(
                time=bottle_start, initial_size=Ne * bottle_mag, population_id=0
            ),
            msp.PopulationParametersChange(
                time=bottle_start + bottle_duration, initial_size=Ne, population_id=0
            ),
        ]


class SerialDivergence(Simulation):
    """Pants model of divergence with one sample as ancient."""

    def __init__(self, Ne, t_div, mod_n, t_anc, n_anc, eps=1e-8):
        """Class defining two-population serial coalescent with divergence.

        one population contains all of the modern samples and a
        second diverged population contains the ancient samples.

        """
        super().__init__()

        # Define effective population size
        self.Ne = Ne

        # Setup samples here
        assert len(t_anc) == len(n_anc)

        samples = [msp.Sample(population=0, time=0) for i in range(mod_n)]
        for (t, n) in zip(t_anc, n_anc):
            for i in range(n):
                # Append ancient samples in the other population
                samples.append(msp.Sample(population=1, time=t))

        self.samples = samples

        # Define population configuration
        self.pop_config = [msp.PopulationConfiguration(), msp.PopulationConfiguration()]

        # Define a mass migration of all the lineages post-split
        self.demography = [
            msp.MassMigration(time=np.max(t_anc) + t_div + eps, source=1, dest=0)
        ]


class SerialMigration(Simulation):
    """Coalescent model of two demes with migration and serial sampling."""

    def __init__(self, Ne, mod_n, t_anc, n_anc, mig_rate=1e-3, eps=1e-8):
        """Two-population serial coalescent with migration between two demes."""
        super().__init__()

        # Define effective population size
        self.Ne = Ne

        # Setup samples here
        assert len(t_anc) == len(n_anc)

        samples = [msp.Sample(population=0, time=0) for i in range(mod_n)]
        for (t, n) in zip(t_anc, n_anc):
            for _ in range(n):
                # Append ancient samples in the other population
                samples.append(msp.Sample(population=1, time=t))

        self.samples = samples

        # Define population configuration
        self.pop_config = [msp.PopulationConfiguration(), msp.PopulationConfiguration()]

        # Setting up the migration matrix
        self.migration_matrix = [[0, mig_rate], [mig_rate, 0]]

    def _simulate(self, **kwargs):
        """Conduct a simulation with msprime class parameters."""
        tree_seq = msp.simulate(
            Ne=self.Ne,
            samples=self.samples,
            migration_matrix=self.migration_matrix,
            population_configurations=self.pop_config,
            **kwargs
        )
        return tree_seq


class SerialTennessenModel(Simulation):
    """The Tennessen et al European Demography from the stdPopSim Consortium.

    https://github.com/popsim-consortium/stdpopsim/blob/c8b557bbfb38ad4371a818dc30acf0e65f15e182/stdpopsim/catalog/homo_sapiens.py#L304

    """

    def __init__(self):
        """Initialize the model with fixed parameters."""
        super().__init__()
        self.Ne = 1e4
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

    def _add_samples(self, mod_pop=1, anc_pop=1, n_mod=100, n_anc=[1], t_anc=[10]):
        """Adding samples to the simulation.

        Default is to have both modern and ancient samples be European.

        """
        assert len(t_anc) == len(n_anc)  # need to make sure these match up
        samples = [msp.Sample(mod_pop, 0) for i in range(n_mod)]
        for (t, n) in zip(t_anc, n_anc):
            for i in range(n):
                # Append samples in the same
                samples.append(msp.Sample(anc_pop, t))

        # Set samples as the parameters for the object
        self.samples = samples

    def _simulate(self, **kwargs):
        """Simulate a panel of individuals under the Tennessen et al European Model."""
        assert self.samples is not None
        # Generate a tree sequence
        tree_seq = msp.simulate(
            samples=self.samples,
            demographic_events=self.demography,
            population_configurations=self.pop_config,
            migration_matrix=self.migration_matrix,
            **kwargs
        )
        return tree_seq

    def _change_growth_rate(self, r_EU=0.05):
        self.Ne = 1e4
        generation_time = 25
        T_AF = 148e3 / generation_time
        T_OOA = 51e3 / generation_time
        T_EU0 = 23e3 / generation_time
        T_EG = 5115 / generation_time

        # Growth rates
        r_EU0 = 0.00307
        # NOTE : this is the recent growth we are changing uplooking at here ...
        #       r_EU = 0.0195
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


class SerialGrowth(Simulation):
    """Simple model of expoential growth with serial sampling."""

    def __init__(self, r=0.1, T=100):
        """Initialize the model.

        Params:
            * r - exponential growth rate
            * T - onset of growth

        """
        super().__init__()
        self.Ne = 1e4

        Ne0 = self.Ne / np.exp(-r * T)

        population_configurations = [
            msp.PopulationConfiguration(initial_size=Ne0, growth_rate=r)
        ]

        self.pop_config = population_configurations

    def _add_samples(self, n_mod=100, n_anc=[1], t_anc=[10]):
        """Adding in the samples."""
        assert len(t_anc) == len(n_anc)  # need to make sure these match up
        samples = [msp.Sample(0, 0) for i in range(n_mod)]
        for (t, n) in zip(t_anc, n_anc):
            for i in range(n):
                # Append samples in the same
                samples.append(msp.Sample(0, t))

        # Set samples as the parameters for the object
        self.samples = samples

    def _simulate(self, **kwargs):
        """Simulate a panel of individuals under the population growth model."""
        assert self.samples is not None
        # Generate a tree sequence
        tree_seq = msp.simulate(
            samples=self.samples, population_configurations=self.pop_config, **kwargs
        )
        return tree_seq


class SerialGrowthSimple(Simulation):
    """Simple serial growth model of exponential growth."""

    def __init__(self, r=0.1, N0=1e6):
        """Initialize the model."""
        super().__init__()
        self.Ne = 1e3

        population_configurations = [
            msp.PopulationConfiguration(initial_size=N0, growth_rate=r)
        ]

        self.pop_config = population_configurations

    def _add_samples(self, n_mod=100, n_anc=[1], t_anc=[10]):
        """Add in samples."""
        assert len(t_anc) == len(n_anc)  # need to make sure these match up
        samples = [msp.Sample(0, 0) for i in range(n_mod)]
        for (t, n) in zip(t_anc, n_anc):
            for i in range(n):
                # Append samples in the same
                samples.append(msp.Sample(0, t))

        # Set samples as the parameters for the object
        self.samples = samples

    def _simulate(self, **kwargs):
        """Simulate a panel of individuals under the population growth model."""
        assert self.samples is not None
        # Generate a tree sequence
        tree_seq = msp.simulate(
            samples=self.samples, population_configurations=self.pop_config, **kwargs
        )
        return tree_seq


class SerialIBDNeUK10K(Simulation):
    """Simulation using demography estimated via IBDNe (UK10K)."""

    def __init__(self, demo_file=None):
        """Initialize the model."""
        super().__init__()
        self.Ne = 1e4
        assert demo_file is not None
        self.demo_file = demo_file

    def _set_demography(self):
        """Establish the demography."""
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

        # Setting the population configurations / demography
        self.pop_config = [msp.PopulationConfiguration(initial_size=N0)]
        self.demography = demography

    def _add_samples(self, n_mod=100, n_anc=[1], t_anc=[10]):
        """Add in the samples."""
        assert len(t_anc) == len(n_anc)  # need to make sure these match up
        samples = [msp.Sample(0, 0) for i in range(n_mod)]
        for (t, n) in zip(t_anc, n_anc):
            for i in range(n):
                # Append samples in the same
                samples.append(msp.Sample(0, t))

        # Set samples as the parameters for the object
        self.samples = samples


class coalSimUtils:
    """Utilities for coalescent simulations."""

    def filt_time_points(ts, ti, **kwargs):
        """Filter to samples at a particular point in time.

        Arguments
        ---------
        ts : msprime.TreeSequence
            tree sequence object

        ti : float
            time-point for sampling

        Returns
        -------
        ts : msprime.TreeSequence
            modified tree-sequence structure with ascertainment

        """
        sample_list = []
        t = ts.first()
        for n in t.samples():
            if t.time(n) == ti:
                sample_list.append(n)
        return ts.simplify(samples=sample_list, **kwargs)

    def ascertain_modern(ts, daf=0.0, **kwargs):
        """Ascertains mutations to sites that are segregating in modern samples.

        Arguments
        ---------
        ts : msprime.TreeSequence
            tree sequence object

        daf : float
            minimum derived allele frequency for sites in the modern individuals

        Returns
        -------
        ts : msprime.TreeSequence
            modified tree-sequence structure with ascertainment

        """
        modern_ts = coalSimUtils.filt_time_points(ts, 0.0, **kwargs)
        n_mod = float(modern_ts.num_samples)
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        for tree in ts.trees():
            for site in tree.sites():
                assert len(site.mutations) == 1  # Only supports infinite sites muts.
                mut = site.mutations[0]
                mod_carriers = [
                    i for i in tree.samples(mut.node) if tree.time(i) == 0.0
                ]
                f = len(mod_carriers) / n_mod
                if (f > daf) & (f < 1.0):
                    site_id = tables.sites.add_row(
                        position=site.position, ancestral_state=site.ancestral_state
                    )
                    tables.mutations.add_row(
                        site=site_id, node=mut.node, derived_state=mut.derived_state
                    )
        return tables.tree_sequence()

    def ascertain_seg_both(ts, t1, daf=0.0, **kwargs):
        """Ascertain sites with mutations segregating at both time-points.

        Arguments
        ---------
        ts : msprime.TreeSequence
            tree sequence object from simulation

        t1 : float
            time-point to filter mutations to

        daf : float
            filter to variants with DAF > daf in both time-points

        Returns
        -------
        ts : msprime.TreeSequence
            modified tree-sequence structure with ascertainment

        """
        # 1. Filter to individuals at modern and at t1
        t = ts.first()
        samples_lst = [n for n in ts.samples() if np.isin(t.time(n), [0.0, t1])]
        t_pts = np.array([t.time(n) for n in ts.samples()])
        n_mod_samp = np.sum(t_pts == 0.0)
        n_t1_samp = np.sum(t_pts == t1)
        filt_ts = ts.simplify(samples=samples_lst, **kwargs)
        # Clear the tables
        tables = filt_ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()

        # 2. Go through tree and filter mutations by criteria
        for tree in filt_ts.trees():
            for site in tree.sites():
                assert len(site.mutations) == 1  # only support infinite-sites mutations
                mut = site.mutations[0]
                tpts = np.array([tree.time(s) for s in tree.samples(mut.node)])
                f_mod = float(np.sum(tpts == 0.0) / n_mod_samp)
                f_t1 = float(np.sum(tpts == t1) / n_t1_samp)
                # Check that it is truly polymorphic at both time-points
                if ((f_mod > daf) & (f_mod < 1.0)) and ((f_t1 > daf) & (f_t1 < 1.0)):
                    site_id = tables.sites.add_row(
                        position=site.position, ancestral_state=site.ancestral_state
                    )
                    tables.mutations.add_row(
                        site=site_id, node=mut.node, derived_state=mut.derived_state
                    )
        return tables.tree_sequence()

    def pos_r2(ts):
        """Obtain vectors of position differences and r^2 per pair of sites.

        Arguments
        ---------
        ts : msprime.TreeSequence
            tree sequence object

        Returns
        -------
        pos_diff : np.array
            position difference for pairs of snps

        r2 : np.array
            r^2 as computed between the different sites

        """
        ld_calc = msp.LdCalculator(ts)
        r2_est = ld_calc.r2_matrix()
        # Computing positions and indices
        pos = np.array([s.position for s in ts.sites()], dtype=np.float32)
        n_sites = ts.num_sites
        pos_diff_mat = np.zeros(shape=(n_sites, n_sites), dtype=np.float32)
        #       print(r2_est.shape, pos_diff_mat.shape)
        for i in np.arange(len(pos)):
            for j in np.arange(i):
                # Calculating the absolute difference in position
                pos_diff_mat[i, j] = np.abs(pos[i] - pos[j])

        # Extract entries that matter (and are matched)
        r2 = r2_est[pos_diff_mat > 0]
        pos_diff = pos_diff_mat[pos_diff_mat > 0]
        return (pos_diff, r2)
