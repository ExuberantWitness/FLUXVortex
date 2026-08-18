#!/usr/bin/env julia

"""
Export deterministic FLOWVPM.jl Float64 reference fixtures for the isolated
FluxV v5h Python parity implementation.

This script intentionally uses direct interactions, the Gaussian-erf kernel,
inviscid rVPM, no SFS, and no relaxation during RK integration.  It does not
read any aerodynamic experiment or FluxV target-case result.
"""

import FLOWVPM
import HDF5
import JSON
import LinearAlgebra
import SHA

const FLOWVPM_COMMIT = "4f433fb09f6baad25db65c9905e0d9cbb09663ce"
const FLOWVPM_TREE = "ecb0fc0b7f7cda244cef695ff06ce23719ad1920"
const FASTMULTIPOLE_COMMIT = "adc4f26"
const FASTMULTIPOLE_TREE = "313cf60bed67629b1da6fb94b3b25394bd4f51ec"
const SCHEMA_VERSION = "flowvpm_oracle_v2"
const J_NAMES = (
    "j11", "j21", "j31",
    "j12", "j22", "j32",
    "j13", "j23", "j33",
)

function sha256_file(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function assert_environment()
    Threads.nthreads() == 1 || error("oracle requires JULIA_NUM_THREADS=1")
    LinearAlgebra.BLAS.set_num_threads(1)
    pkgversion(FLOWVPM) == v"4.0.4" || error("unexpected FLOWVPM version")

    manifest_path = joinpath(@__DIR__, "Manifest.toml")
    manifest = read(manifest_path, String)
    occursin("repo-rev = \"$(FLOWVPM_COMMIT)\"", manifest) ||
        error("FLOWVPM commit is not pinned in Manifest.toml")
    occursin("git-tree-sha1 = \"$(FLOWVPM_TREE)\"", manifest) ||
        error("FLOWVPM tree is not pinned in Manifest.toml")
    occursin("repo-rev = \"$(FASTMULTIPOLE_COMMIT)\"", manifest) ||
        error("FastMultipole commit is not pinned in Manifest.toml")
    occursin("git-tree-sha1 = \"$(FASTMULTIPOLE_TREE)\"", manifest) ||
        error("FastMultipole tree is not pinned in Manifest.toml")
end

function new_field(
    maxparticles::Int;
    uinf=(0.0, 0.0, 0.0),
    uinf_function=nothing,
)
    uinf_vector = collect(Float64, uinf)
    uinf_provider = if isnothing(uinf_function)
        (t) -> uinf_vector
    else
        uinf_function
    end
    return FLOWVPM.ParticleField(
        maxparticles,
        Float64;
        formulation=FLOWVPM.rVPM,
        viscous=FLOWVPM.Inviscid(),
        transposed=true,
        SFS=FLOWVPM.noSFS,
        kernel=FLOWVPM.gaussianerf,
        UJ=FLOWVPM.UJ_direct,
        Uinf=uinf_provider,
        relaxation=FLOWVPM.norelaxation,
        integration=FLOWVPM.rungekutta3,
        useGPU=0,
    )
end

function add_fixture_particles!(pfield, x, gamma, sigma)
    size(x, 2) == 3 || error("x must be N x 3")
    size(gamma) == size(x) || error("gamma must match x")
    length(sigma) == size(x, 1) || error("sigma must have length N")
    all(isfinite, x) || error("x contains non-finite values")
    all(isfinite, gamma) || error("gamma contains non-finite values")
    all(isfinite, sigma) || error("sigma contains non-finite values")
    all(>(0.0), sigma) || error("sigma must be positive")

    for i in axes(x, 1)
        FLOWVPM.add_particle(
            pfield,
            view(x, i, :),
            view(gamma, i, :),
            sigma[i];
            vol=0.0,
            circulation=1.0,
            C=0.0,
            static=false,
        )
    end
    return pfield
end

function fixture_arrays(; probes=true)
    x_source = [
        0.00 0.00 0.00
        0.31 -0.14 0.22
        -0.27 0.38 -0.19
        0.18 0.29 0.41
        -0.42 -0.23 0.17
    ]
    gamma_source = [
        0.12 -0.07 0.05
        -0.03 0.11 0.08
        0.09 0.04 -0.06
        -0.08 0.02 0.13
        0.05 -0.12 0.01
    ]
    sigma_source = [0.16, 0.21, 0.13, 0.19, 0.17]

    if !probes
        return x_source, gamma_source, sigma_source, ones(Int64, 5)
    end

    x_probe = [
        0.07 -0.04 0.03
        1.20 -0.80 0.60
        -0.15 0.11 -0.33
    ]
    gamma_probe = zeros(3, 3)
    sigma_probe = [0.15, 0.24, 0.18]
    return (
        vcat(x_source, x_probe),
        vcat(gamma_source, gamma_probe),
        vcat(sigma_source, sigma_probe),
        vcat(ones(Int64, 5), zeros(Int64, 3)),
    )
end

function snapshot(pfield)
    n = FLOWVPM.get_np(pfield)
    particles = view(pfield.particles, :, 1:n)
    return (
        x=permutedims(Array(view(particles, 1:3, :))),
        gamma=permutedims(Array(view(particles, 4:6, :))),
        sigma=vec(Array(view(particles, 7, :))),
        u=permutedims(Array(view(particles, 10:12, :))),
        j=permutedims(Array(view(particles, 16:24, :))),
        m=permutedims(Array(view(particles, 28:36, :))),
    )
end

function write_vector_components!(group, prefix::AbstractString, values)
    size(values, 2) == 3 || error("$prefix must be N x 3")
    HDF5.write(group, "$(prefix)_x", values[:, 1])
    HDF5.write(group, "$(prefix)_y", values[:, 2])
    HDF5.write(group, "$(prefix)_z", values[:, 3])
end

function write_state!(group, state)
    write_vector_components!(group, "x", state.x)
    write_vector_components!(group, "gamma", state.gamma)
    HDF5.write(group, "sigma", state.sigma)
    for i in 1:9
        HDF5.write(group, "m$(lpad(i, 2, '0'))", state.m[:, i])
    end
end

function write_field!(group, state)
    write_vector_components!(group, "u", state.u)
    for (i, name) in enumerate(J_NAMES)
        HDF5.write(group, name, state.j[:, i])
    end
end

function write_input!(group, x, gamma, sigma; role_code=nothing)
    write_vector_components!(group, "x", x)
    write_vector_components!(group, "gamma", gamma)
    HDF5.write(group, "sigma", sigma)
    HDF5.write(group, "particle_id", collect(Int64, axes(x, 1)))
    if !isnothing(role_code)
        HDF5.write(group, "role_code", role_code)
    end
end

function run_uj_fixture!(root)
    fixture = HDF5.create_group(root, "uj_direct_gauserf")
    x, gamma, sigma, role = fixture_arrays(; probes=true)
    pfield = new_field(size(x, 1))
    add_fixture_particles!(pfield, x, gamma, sigma)
    FLOWVPM.UJ_direct(pfield; reset=true, reset_sfs=false, sfs=false)

    input_group = HDF5.create_group(fixture, "input")
    output_group = HDF5.create_group(fixture, "output")
    write_input!(input_group, x, gamma, sigma; role_code=role)
    HDF5.write(input_group, "role_code_meaning", "1=source,0=zero-strength probe")
    write_field!(output_group, snapshot(pfield))
end

function run_nearfield_fixture!(root)
    fixture = HDF5.create_group(root, "uj_direct_gauserf_nearfield_sweep")
    source_sigma = 0.2
    ratios = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    raw_directions = [
        1.0 2.0 3.0
        -2.0 1.0 0.5
        0.3 -1.0 2.0
        1.0 0.2 -0.7
        -0.4 0.9 1.0
        0.8 -0.6 0.5
        -1.0 -0.3 0.8
        0.2 1.0 -0.4
        0.7 0.4 1.0
    ]
    directions = copy(raw_directions)
    for i in axes(directions, 1)
        directions[i, :] ./= LinearAlgebra.norm(view(directions, i, :))
    end
    x_probe = directions .* reshape(source_sigma .* ratios, :, 1)
    x = vcat(zeros(1, 3), x_probe)
    gamma = vcat(reshape([0.13, -0.09, 0.07], 1, 3), zeros(length(ratios), 3))
    sigma = fill(source_sigma, size(x, 1))
    role = vcat(Int64[1], zeros(Int64, length(ratios)))

    pfield = new_field(size(x, 1))
    add_fixture_particles!(pfield, x, gamma, sigma)
    FLOWVPM.UJ_direct(pfield; reset=true, reset_sfs=false, sfs=false)

    input_group = HDF5.create_group(fixture, "input")
    output_group = HDF5.create_group(fixture, "output")
    write_input!(input_group, x, gamma, sigma; role_code=role)
    HDF5.write(input_group, "role_code_meaning", "1=single source,0=zero-strength probe")
    HDF5.write(input_group, "source_particle_id", Int64[1])
    HDF5.write(input_group, "probe_particle_id", collect(Int64, 2:size(x, 1)))
    HDF5.write(input_group, "r_over_source_sigma", ratios)
    write_field!(output_group, snapshot(pfield))
end

function write_rk_config!(config, dt, steps)
    HDF5.write(config, "dt", [dt])
    HDF5.write(config, "steps", Int64[steps])
    HDF5.write(config, "rk_a", [0.0, -5 / 9, -153 / 128])
    HDF5.write(config, "rk_b", [1 / 3, 15 / 16, 8 / 15])
    HDF5.write(config, "transposed", Int64[1])
    HDF5.write(config, "formulation_f", [0.0])
    HDF5.write(config, "formulation_g", [0.2])
    HDF5.write(config, "sfs_enabled", Int64[0])
    HDF5.write(config, "viscosity_enabled", Int64[0])
    HDF5.write(config, "relaxation_enabled", Int64[0])
    HDF5.write(config, "stage_schema", "pre=state_only;rhs=UJ_at_pre;post=state_only")
end

function recorded_rk_step!(pfield, dt)
    pre = Any[]
    rhs = Any[]
    function recorded_uj(field; kwargs...)
        push!(pre, snapshot(field))
        FLOWVPM.UJ_direct(field; kwargs...)
        push!(rhs, snapshot(field))
        return nothing
    end

    FLOWVPM.nextstep(
        pfield,
        dt;
        relax=false,
        custom_UJ=recorded_uj,
    )
    length(pre) == 3 || error("RK3 did not call UJ three times")
    length(rhs) == 3 || error("RK3 did not record three RHS evaluations")
    return pre, rhs, (pre[2], pre[3], snapshot(pfield))
end

function write_rk_step!(
    fixture,
    step,
    pfield,
    pre,
    rhs,
    post;
    field_time_before,
    uinf_used,
)
    step_group = HDF5.create_group(fixture, "step_$(lpad(step, 2, '0'))")
    HDF5.write(step_group, "field_time_before", [Float64(field_time_before)])
    HDF5.write(step_group, "uinf_used_x", [Float64(uinf_used[1])])
    HDF5.write(step_group, "uinf_used_y", [Float64(uinf_used[2])])
    HDF5.write(step_group, "uinf_used_z", [Float64(uinf_used[3])])
    for stage in 1:3
        stage_group = HDF5.create_group(
            step_group,
            "stage_$(lpad(stage, 2, '0'))",
        )
        write_state!(HDF5.create_group(stage_group, "pre"), pre[stage])
        write_field!(HDF5.create_group(stage_group, "rhs"), rhs[stage])
        write_state!(HDF5.create_group(stage_group, "post"), post[stage])
    end
    HDF5.write(step_group, "field_time_after", [Float64(pfield.t)])
    HDF5.write(step_group, "field_step_after", Int64[pfield.nt])
end

function run_rk_fixture!(root)
    fixture = HDF5.create_group(root, "rk3_rvpm_direct_gauserf")
    x, gamma, sigma, _ = fixture_arrays(; probes=false)
    dt = 0.0125
    steps = 2
    uinf = [0.17, -0.04, 0.09]
    pfield = new_field(size(x, 1); uinf=uinf)
    add_fixture_particles!(pfield, x, gamma, sigma)

    config = HDF5.create_group(fixture, "config")
    write_rk_config!(config, dt, steps)
    HDF5.write(config, "uinf_x", [uinf[1]])
    HDF5.write(config, "uinf_y", [uinf[2]])
    HDF5.write(config, "uinf_z", [uinf[3]])
    HDF5.write(config, "uinf_evaluation_contract", "constant_vector_once_per_step")

    input_group = HDF5.create_group(fixture, "input")
    write_input!(input_group, x, gamma, sigma)

    for step in 1:steps
        field_time_before = pfield.t
        pre, rhs, post = recorded_rk_step!(pfield, dt)
        write_rk_step!(
            fixture,
            step,
            pfield,
            pre,
            rhs,
            post;
            field_time_before=field_time_before,
            uinf_used=uinf,
        )
    end
end

function run_timevarying_uinf_fixture!(root)
    fixture = HDF5.create_group(root, "rk3_timevarying_uinf_direct_gauserf")
    x = reshape([0.11, -0.07, 0.03], 1, 3)
    gamma = reshape([0.21, -0.08, 0.13], 1, 3)
    sigma = [0.18]
    dt = 0.02
    steps = 2
    uinf_base = [0.12, -0.03, 0.08]
    uinf_slope = [0.70, 0.20, -0.40]
    uinf_function = (t) -> uinf_base .+ uinf_slope .* t
    pfield = new_field(size(x, 1); uinf_function=uinf_function)
    add_fixture_particles!(pfield, x, gamma, sigma)

    config = HDF5.create_group(fixture, "config")
    write_rk_config!(config, dt, steps)
    for (i, axis) in enumerate(("x", "y", "z"))
        HDF5.write(config, "uinf_base_$axis", [uinf_base[i]])
        HDF5.write(config, "uinf_slope_$axis", [uinf_slope[i]])
    end
    HDF5.write(config, "uinf_model", "affine_in_field_time")
    HDF5.write(
        config,
        "uinf_evaluation_contract",
        "once_per_step_at_field_time_before_rk_stages",
    )

    input_group = HDF5.create_group(fixture, "input")
    write_input!(input_group, x, gamma, sigma)

    for step in 1:steps
        field_time_before = pfield.t
        uinf_used = uinf_function(field_time_before)
        pre, rhs, post = recorded_rk_step!(pfield, dt)
        write_rk_step!(
            fixture,
            step,
            pfield,
            pre,
            rhs,
            post;
            field_time_before=field_time_before,
            uinf_used=uinf_used,
        )
    end
end

function make_relaxation_j(w; diagonal=(0.1, -0.2, 0.3))
    # FLOWVPM column-major storage gives
    # w = (J6-J8, J7-J3, J2-J4).
    return [
        diagonal[1], w[3], 0.0,
        0.0, diagonal[2], w[1],
        w[2], 0.0, diagonal[3],
    ]
end

function run_relaxation_fixture!(root)
    fixture = HDF5.create_group(root, "corrected_pedrizzetti")
    gamma = [0.31, -0.22, 0.17]
    w = [0.20, -0.40, 0.70]
    cases = (
        ("case_001", 0.0, gamma, make_relaxation_j(w), "nonparallel_alpha_0"),
        ("case_002", 0.3, gamma, make_relaxation_j(w), "nonparallel_alpha_0p3"),
        ("case_003", 1.0, gamma, make_relaxation_j(w), "nonparallel_alpha_1"),
        (
            "case_004",
            0.3,
            gamma,
            make_relaxation_j([0.0, 0.0, 0.0]),
            "zero_vorticity_unchanged",
        ),
    )

    for (case_name, alpha, gamma_input, j_input, description) in cases
        pfield = new_field(1)
        FLOWVPM.add_particle(
            pfield,
            zeros(3),
            gamma_input,
            0.2;
            vol=0.0,
            circulation=1.0,
            C=0.0,
            static=false,
        )
        FLOWVPM.get_J(pfield, 1) .= j_input
        gamma_before = copy(FLOWVPM.get_Gamma(pfield, 1))
        FLOWVPM.relax_correctedpedrizzetti(alpha, pfield, 1)
        gamma_after = copy(FLOWVPM.get_Gamma(pfield, 1))

        case_group = HDF5.create_group(fixture, case_name)
        HDF5.write(case_group, "description", description)
        HDF5.write(case_group, "alpha", [alpha])
        write_vector_components!(case_group, "gamma_before", reshape(gamma_before, 1, 3))
        HDF5.write(case_group, "j_column_major", j_input)
        write_vector_components!(case_group, "gamma_after", reshape(gamma_after, 1, 3))
        HDF5.write(case_group, "norm_before", [LinearAlgebra.norm(gamma_before)])
        HDF5.write(case_group, "norm_after", [LinearAlgebra.norm(gamma_after)])
    end
    HDF5.write(
        fixture,
        "invalid_contract",
        "Gamma=0 with nonzero vorticity is excluded upstream and must fail in Python",
    )
end

function write_metadata!(file, output_path)
    meta = HDF5.create_group(file, "meta")
    HDF5.write(meta, "schema_version", SCHEMA_VERSION)
    HDF5.write(meta, "source_commit", FLOWVPM_COMMIT)
    HDF5.write(meta, "source_tree", FLOWVPM_TREE)
    HDF5.write(meta, "fastmultipole_commit", FASTMULTIPOLE_COMMIT)
    HDF5.write(meta, "fastmultipole_tree", FASTMULTIPOLE_TREE)
    HDF5.write(meta, "flowvpm_version", string(pkgversion(FLOWVPM)))
    HDF5.write(meta, "hdf5_version", string(pkgversion(HDF5)))
    HDF5.write(meta, "julia_version", string(VERSION))
    HDF5.write(meta, "julia_threads", Int64[Threads.nthreads()])
    HDF5.write(meta, "blas_threads", Int64[LinearAlgebra.BLAS.get_num_threads()])
    HDF5.write(meta, "machine", Sys.MACHINE)
    HDF5.write(meta, "project_sha256", sha256_file(joinpath(@__DIR__, "Project.toml")))
    HDF5.write(meta, "manifest_sha256", sha256_file(joinpath(@__DIR__, "Manifest.toml")))
    HDF5.write(meta, "export_script_sha256", sha256_file(@__FILE__))
    HDF5.write(meta, "output_basename", basename(output_path))
    HDF5.write(meta, "json_mirror_basename", replace(basename(output_path), r"\.h5$" => ".json"))
    HDF5.write(meta, "j_storage", join(J_NAMES, ","))
    HDF5.write(meta, "stretching_convention", "transposed=true; S=J^T*Gamma")
    HDF5.write(meta, "interaction_mode", "direct_float64_self_interaction_excluded")
    HDF5.write(meta, "stage_state_schema", "pre/state_only;rhs/UJ_only;post/state_only")
    HDF5.write(meta, "rhs_evaluation_state", "stage_pre")
end

function hdf5_group_to_dict(group)
    result = Dict{String,Any}()
    for name in keys(group)
        child = group[name]
        result[String(name)] = if child isa HDF5.Group
            hdf5_group_to_dict(child)
        else
            read(child)
        end
    end
    return result
end

function write_json_mirror(hdf5_path, json_path)
    payload = HDF5.h5open(hdf5_path, "r") do file
        hdf5_group_to_dict(file)
    end
    open(json_path, "w") do io
        JSON.print(io, payload, 2)
        write(io, '\n')
    end
end

function parse_output(args)
    length(args) == 2 || error("usage: export_oracle.jl --output PATH")
    args[1] == "--output" || error("first argument must be --output")
    isempty(args[2]) && error("output path cannot be empty")
    return abspath(args[2])
end

function main(args=ARGS)
    assert_environment()
    output_path = parse_output(args)
    endswith(output_path, ".h5") || error("output path must end in .h5")
    json_path = replace(output_path, r"\.h5$" => ".json")
    ispath(output_path) && error("refusing to overwrite existing output: $output_path")
    ispath(json_path) && error("refusing to overwrite existing output: $json_path")
    mkpath(dirname(output_path))

    HDF5.h5open(output_path, "w") do file
        write_metadata!(file, output_path)
        fixtures = HDF5.create_group(file, "fixtures")
        run_uj_fixture!(fixtures)
        run_nearfield_fixture!(fixtures)
        run_rk_fixture!(fixtures)
        run_timevarying_uinf_fixture!(fixtures)
        run_relaxation_fixture!(fixtures)
    end
    write_json_mirror(output_path, json_path)
    println("wrote ", output_path)
    println("sha256 ", sha256_file(output_path))
    println("wrote ", json_path)
    println("sha256 ", sha256_file(json_path))
end

main()
