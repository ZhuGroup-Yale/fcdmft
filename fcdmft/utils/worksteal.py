from mpi4py import MPI
import numpy as np


class MPIWorkStealingScheduler:
    def __init__(self, cartcomm=None):
        """MPI 3 RMA-based work stealing scheduler

        Parameters
        ----------
        cartcomm : mpi4py.MPI.Cartcomm, optional
            1-D Cartesian communicator
        """
        if cartcomm is None:
            commworld = MPI.COMM_WORLD
            cartcomm = commworld.Create_cart(
                dims=[commworld.Get_size()], periods=[True], reorder=True
            )

        assert cartcomm.Get_dim() == 1, "Cartcomm must be 1-D"
        self.cartcomm = cartcomm

        self.workitems = None
        self.nwork_by_rank = None
        self.cur_job_win = None
        self.cur_job_local = None
        self.global_workitems = None
        self.one = np.array([1], dtype=np.int64)
        self.buf_one = np.zeros(1, dtype=np.int64)
        self.cur_victim = cartcomm.Get_rank()
        self.empty = False

    def initialize_workitems(self, workitems):
        """Put jobs in the scheduler.
        
        A window is then created on each rank to store the current job index.

        Parameters
        ----------
        workitems : array_like of int
            2D array (or list of tuples) of jobs, one per row.

        """
        self.workitems = np.asarray(workitems, dtype=np.int64)
        self.global_workitems = self.cartcomm.allgather(self.workitems)

        self.nwork_by_rank = [w.shape[0] for w in self.global_workitems]
        self.cur_job_win = MPI.Win.Allocate(size=MPI.INT64_T.size, comm=self.cartcomm)
        self.cur_job_local = np.frombuffer(self.cur_job_win.tomemory(), dtype=np.int64)
        self.cur_job_win.Lock_all()
        self.cur_job_local[0] = 0

        self.cur_job_win.Sync()

    def finalize(self):
        """Free the RMA window and clean up.
        """
        self.cur_job_win.Unlock_all()
        self.cur_job_win.Free()
        self.global_workitems = None
        self.workitems = None

    def get_workitems(self):
        """Generator that yields jobs to be done.

        The main work-stealing loop lives here.

        Yields
        ------
        ndarray
            1D array of np.int64; a row of the global workitems array.
        """
        while True:
            # try to get a job from the victim's queue
            self.cur_job_win.Get_accumulate(
                [self.one, 1, MPI.INT64_T],
                [self.buf_one, 1, MPI.INT64_T],
                target_rank=self.cur_victim,
            )
            self.cur_job_win.Flush_local(self.cur_victim)
            job_idx = self.buf_one[0]

            # victim's queue has a job. steal it!
            if job_idx < self.nwork_by_rank[self.cur_victim]:
                # steal job:
                yield self.global_workitems[self.cur_victim][job_idx]
            # victim's queue is empty. try next victim
            else:
                self.cur_victim = (self.cur_victim + 1) % self.cartcomm.size
                # wrapped all the way around. Done!
                if self.cur_victim == self.cartcomm.rank:
                    break


if __name__ == "__main__":
    import sys

    scheduler = MPIWorkStealingScheduler()
    rank = scheduler.cartcomm.Get_rank()
    size = scheduler.cartcomm.Get_size()
    nworkperrank = 40

    workprefactor = 1000

    loc_workitems = [
        (i + nworkperrank * rank, np.random.randint(workprefactor))
        for i in range(nworkperrank)
    ]
    glob_workitems = scheduler.cartcomm.allgather(loc_workitems)

    scheduler.initialize_workitems(np.asarray(loc_workitems))
    print("Initialized work items.")
    local_done = []
    for idx, size in scheduler.get_workitems():
        x = np.random.random((size, size))
        y = x @ x.T + x.T @ x
        del x
        del y
        local_done.append(idx)

    global_done = scheduler.cartcomm.allgather(local_done)
    done = np.sort(np.concatenate(global_done, axis=0))
    sys.stdout.flush()
    scheduler.cartcomm.barrier()
    print(f"Rank {rank} did {len(local_done)} jobs")
    sys.stdout.flush()
    scheduler.cartcomm.barrier()
    if rank == 0:
        print(f"{len(done)} jobs done out of {nworkperrank*scheduler.cartcomm.size}")
        uniq, cts = np.unique(done, return_counts=True)
        print(f"{uniq.size} of these jobs unique")
        if uniq.size == nworkperrank * scheduler.cartcomm.size:
            print("All jobs done")
    scheduler.finalize()
    MPI.Finalize()
