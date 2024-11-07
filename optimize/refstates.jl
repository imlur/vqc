# function related to initial setting (running DMRG and saving the result)
using HDF5, ITensorMPS

function buildhamiltonian(L::Int, ii::Index)
	os = OpSum()
	for i=1:L-1
		os += 1/2,"S+",i,"S-",i+1
		os += 1/2,"S-",i,"S+",i+1
		os += "Sz",i,"Sz",i+1
	end
	sites = [settags(ii, "S=1/2,site,s$(i)") for i=1:L]
	removetags(MPO(os, sites), "S=1/2")
end

sites(L::Int, ii::Index) = [settags(ii, "site,s$(i)") for i=1:L]

fname(L::Int, maxdim::Int, nsweeps::Int) = 
	"$(dirname(@__FILE__))/../dmrgresult/result_L$(L)_d$(maxdim)_n$(nsweeps).h5"

rundmrg(L, maxdim, nsweeps=5) = rundmrg(L, Index(2), maxdim, nsweeps)
function rundmrg(L::Int, ii::Index, maxdim::Int, nsweeps=5::Int)
	resfn = fname(L, maxdim, nsweeps)
	if isfile(resfn)
		println("Already exists for L=$(L), maxdim=$(maxdim), nsweeps=$(nsweeps)")
		f = h5open(resfn, "r")
		E, mps = read(f, "E"), read(f, "MPS", MPS)
		close(f)
		return E, mps
	end
	
	H = buildhamiltonian(L::Int, ii::Index)
	psi0 = random_mps(sites(L, ii); linkdims=4)
	# TODO: Figure out what is the appropriate nsweep
	cutoff = 1E-10
	E, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

	f = h5open(resfn, "w")
	write(f, "E", E); write(f, "MPS", psi)
	close(f)
	return E, psi
end
