using HDF5: h5open, read, write
using ITensors: ITensor, scalar
using ITensorMPS: siteinds, state, MPS

function get_mps_coverage(filename, samples_filename, icp, ndt, nstep, nexp, out_filename)
    fp = h5open(filename, "r")
    psi = read(fp, "psi", MPS)
    sidx = siteinds(psi)
    close(fp)

    probs = Array{Float64}(undef, nexp, nstep, ndt)

    fp = h5open(samples_filename, "r")
    for idt in 1:ndt
        for istep in 1:nstep
            for iexp in 1:nexp
                states = read(fp, "c$(icp)/dt$(idt - 1)/step$(istep)/exp$(iexp - 1)") .+ 1
                prob = 0.
                for ist in 1:size(states)[2]
                    V = ITensor(1.)
                    for iq in 1:length(psi)
                        V *= (psi[iq] * state(sidx[iq], states[iq, ist]))
                    end
                    prob += abs(scalar(V)) ^ 2
                end
                probs[iexp, istep, idt] = prob
            end
        end
    end
    close(fp)

    fp = h5open(out_filename, "w")
    write(fp, "probs", probs)
    close(fp)
end

let
    filename = ARGS[1]
    samples_filename = ARGS[2]
    icp = parse(Int, ARGS[3])
    ndt = parse(Int, ARGS[4])
    nstep = parse(Int, ARGS[5])
    nexp = parse(Int, ARGS[6])
    out_filename = ARGS[7]
    get_mps_coverage(filename, samples_filename, icp, ndt, nstep, nexp, out_filename)
end
