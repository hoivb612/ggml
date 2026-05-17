cmake_minimum_required(VERSION 3.19)

if (NOT WIN32)
    message(WARNING "ggml-dx12: DirectX 12 backend is only supported on Windows and Xbox Scarlett (GDKX)")
    return()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_definitions(/MP)
endif()

ggml_add_backend_library(ggml-dx12
    ggml-dx12.cpp
    ../../include/ggml-dx12.h
)

if (GGML_DX12_XBOX)
    # ---- Xbox Series X (Scarlett, GDKX) cross-compile branch ----
    #
    # Output DLL is renamed to ggml-dx12-xbox.dll so it's obvious in the
    # deploy package that this build is the Scarlett-targeted variant, and
    # so it matches the GGML_DX12_XBOX CMake option name.
    set_target_properties(ggml-dx12 PROPERTIES OUTPUT_NAME "ggml-dx12-xbox")

    # _GAMING_XBOX / _GAMING_XBOX_SCARLETT are already set by the toolchain
    # file (cmake/xbox-scarlett.toolchain.cmake); the source code keys all
    # Xbox-specific branches off those

    # GameCore Scarlett system libs. The Gaming.Xbox.Scarlett.x64 MSBuild
    # platform normally injects these via its .props files; list them
    # explicitly so a non-VS generator (Ninja) also links cleanly.
    target_link_libraries(ggml-dx12 PRIVATE
        d3d12_xs
        xg_xs
        xgameruntime
    )

    # GDKX include / lib paths. Honor the toolchain hint first, then the
    # GameDK environment variable, then a manual GGML_DX12_XBOX_GDK_EDITION pin.
    set(_xbox_gdk_root "")
    if (DEFINED GGML_DX12_XBOX_GDK_ROOT AND GGML_DX12_XBOX_GDK_ROOT)
        set(_xbox_gdk_root "${GGML_DX12_XBOX_GDK_ROOT}")
    elseif (DEFINED ENV{GameDK})
        set(_xbox_gdk_root "$ENV{GameDK}")
    endif()

    # Normalize: the GameDK env var from the installer is Windows-style and
    # has a trailing backslash, which breaks file(GLOB ${root}/*).
    if (_xbox_gdk_root)
        file(TO_CMAKE_PATH "${_xbox_gdk_root}" _xbox_gdk_root)
        string(REGEX REPLACE "/+$" "" _xbox_gdk_root "${_xbox_gdk_root}")
    endif()

    if (_xbox_gdk_root)
        if (GGML_DX12_XBOX_GDK_EDITION)
            set(_xbox_gdk_dir "${_xbox_gdk_root}/${GGML_DX12_XBOX_GDK_EDITION}")
        else()
            # Pick the highest-numbered edition subdirectory under %GameDK%.
            file(GLOB _xbox_gdk_eds RELATIVE "${_xbox_gdk_root}" "${_xbox_gdk_root}/*")
            set(_xbox_gdk_versions "")
            foreach (_d ${_xbox_gdk_eds})
                if (_d MATCHES "^[0-9]+$" AND IS_DIRECTORY "${_xbox_gdk_root}/${_d}")
                    list(APPEND _xbox_gdk_versions ${_d})
                endif()
            endforeach()
            if (_xbox_gdk_versions)
                list(SORT _xbox_gdk_versions COMPARE NATURAL ORDER DESCENDING)
                list(GET _xbox_gdk_versions 0 _xbox_gdk_latest)
                set(_xbox_gdk_dir "${_xbox_gdk_root}/${_xbox_gdk_latest}")
            else()
                message(FATAL_ERROR "ggml-dx12 (Xbox): no numeric GDKX edition found under ${_xbox_gdk_root}. Pass -DGGML_DX12_XBOX_GDK_EDITION=<edition>.")
            endif()
        endif()
        message(STATUS "ggml-dx12 (Xbox): using GDKX at ${_xbox_gdk_dir}")
        # The Gaming.Xbox.Scarlett.x64 platform normally adds these include
        # and lib paths automatically. Adding them PRIVATE is a no-op when
        # MSBuild already injected them, but keeps Ninja honest.
        target_include_directories(ggml-dx12 PRIVATE
            "${_xbox_gdk_dir}/GXDK/gameKit/Include"
            "${_xbox_gdk_dir}/GXDK/gameKit/Include/Scarlett"
        )
        target_link_directories(ggml-dx12 PRIVATE
            "${_xbox_gdk_dir}/GXDK/gameKit/Lib/amd64/Scarlett"
            "${_xbox_gdk_dir}/GXDK/gameKit/Lib/amd64"
        )
        set(_xbox_gdk_dir_cached "${_xbox_gdk_dir}" CACHE INTERNAL "Resolved GDKX directory")
    else()
        message(WARNING "ggml-dx12 (Xbox): no GDKX root found. Set %GameDK% or pass -DGGML_DX12_XBOX_GDK_EDITION.")
    endif()
else()
    # Native Windows: link D3D12 and DXGI system libraries
    target_link_libraries(ggml-dx12 PRIVATE
        d3d12
        dxgi
        dxguid
    )
endif()

target_include_directories(ggml-dx12 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Debug options
if (GGML_DX12_DEBUG)
    target_compile_definitions(ggml-dx12 PRIVATE GGML_DX12_DEBUG)
endif()

# HLSL Shader Compilation
if (GGML_DX12_XBOX)
    # ---- Scarlett shader compiler discovery ----
    # User override wins.
    if (GGML_DX12_XBOX_SHADER_COMPILER)
        set(DXC_EXECUTABLE "${GGML_DX12_XBOX_SHADER_COMPILER}" CACHE FILEPATH "" FORCE)
    else()
        # Search the GDKX install for the Scarlett-targeted compiler.
        # Across GDKX revisions the binary has lived under a few names; try
        # the most likely ones in order of preference.
        set(_xbox_dxc_hints "")
        if (DEFINED _xbox_gdk_dir_cached AND _xbox_gdk_dir_cached)
            list(APPEND _xbox_dxc_hints
                "${_xbox_gdk_dir_cached}/GXDK/bin/Scarlett"
                "${_xbox_gdk_dir_cached}/GXDK/bin")
        endif()
        find_program(DXC_EXECUTABLE
            NAMES XboxScarlettShaderCompiler.exe XSCDxc.exe dxc-xs.exe dxc.exe
            HINTS ${_xbox_dxc_hints}
            NO_DEFAULT_PATH)
    endif()

    if (NOT DXC_EXECUTABLE)
        message(FATAL_ERROR
            "ggml-dx12 (Xbox): Scarlett HLSL compiler not found.\n"
            "  Pass -DGGML_DX12_XBOX_SHADER_COMPILER=<path-to-compiler.exe>\n"
            "  or run from a GDKX command prompt so %GameDK% is set.")
    endif()
    message(STATUS "ggml-dx12 (Xbox): Found Scarlett shader compiler: ${DXC_EXECUTABLE}")
else()
    # Find DXC in Windows SDK
    find_program(DXC_EXECUTABLE dxc
        HINTS
            "$ENV{WindowsSdkDir}/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
        "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
        "$ENV{VULKAN_SDK}/Bin"
)
endif()

if (DXC_EXECUTABLE)
    if (NOT GGML_DX12_XBOX)
        message(STATUS "ggml-dx12: Found DXC: ${DXC_EXECUTABLE}")
    endif()

    set(SHADER_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

    # List of shaders to compile
    set(DX12_SHADERS
        add
        mul
        scale
        sqr
        sqrt_
        silu
        gelu
        gelu_quick
        gelu_erf
        relu
        tanh_
        sigmoid
        rms_norm
        norm
        group_norm
        soft_max
        mul_mat
        mul_mat_id
        mul_mat_id_q4_0
        mul_mat_id_q4_1
        mul_mat_id_q5_0
        mul_mat_id_q5_1
        mul_mat_id_q8_0
        mul_mat_id_q4k
        mul_mat_id_q5k
        mul_mat_id_q6k
        mul_mat_id_iq4_nl
        cpy
        get_rows
        diag_mask_inf
        rope
        rope_multi
        clamp
        concat
        repeat
        sum_rows
        argsort
        sub
        div
        sin
        cos
        exp
        softplus
        log
        pad
        roll
        ssm_conv
        ssm_scan
        upscale
        im2col
        pool_2d
        dequant_q4_k
        flash_attn
        flash_attn_64
        flash_attn_128
        flash_attn_gqa
        flash_attn_reduce
        set_rows
        rope_set_rows
        conv_2d
        cast
        pool_1d
        get_rows_q4k
        get_rows_q5k
        get_rows_q6k
        glu
        l2_norm
        gated_delta_net
        rms_norm_mul
        rms_norm_mul_rope
        rms_norm_mul_rope_set_rows
        add_rms_norm_mul
        mul_mat_vec
        mul_mat_vec_mr
        mul_mat_vec_q4k_mr
        mul_mat_vec_q5k_mr
        mul_mat_vec_q5k_subgroup
        mul_mat_vec_q6k_mr
        mul_mat_vec_q5_0_mr
        mul_mat_vec_q5_1_mr
        mul_mat_vec_q8_0_mr
        mul_mat_vec_q8_0_mr256v
        mul_mat_vec_glu
        mul_mat_vec_mr32
        mul_mat_q4_0
        mul_mat_q4_1
        mul_mat_q5_0
        mul_mat_q5_1
        mul_mat_q8_0
        mul_mat_q8_1
        get_rows_q4_0
        get_rows_q4_1
        get_rows_q5_0
        get_rows_q5_1
        get_rows_q8_0
        get_rows_q8_1
        mul_mat_q2k
        mul_mat_q3k
        mul_mat_vec_q2k
        mul_mat_vec_q3k
        mul_mat_vec_q2k_mr
        mul_mat_vec_q3k_mr
        get_rows_q2k
        get_rows_q3k
        mul_mat_iq4_nl
        mul_mat_vec_iq4_nl
        get_rows_iq4_nl
        mul_mat_wmma
        mul_mat_q4k_wmma
        mul_mat_q5k_wmma
        mul_mat_q6k_wmma
        mul_mat_q8_0_wmma
        quantize_q8_1
    )

    # Shaders requiring SM 6.6 (dot4add_i8packed / dp4a — built at 6.6 with rest)
    set(DX12_SHADERS_SM64
        mul_mat_q8_0_q8_1
        mul_mat_vec_q4k_dp4a
        mul_mat_vec_q4k_dp4a_32
        mul_mat_vec_q5k_dp4a
        mul_mat_vec_q5k_dp4a_32
        mul_mat_vec_q6k_dp4a
        mul_mat_vec_q8_0_dp4a
        mul_mat_vec_q5_0_dp4a
        mul_mat_vec_q5_1_dp4a
    )

    set(DX12_SHADER_HEADERS "")

    # Common DXC flags. Shaders that use Inf/NaN-sensitive control flow are
    # listed below and compiled without -ffinite-math-only.
    set(DXC_COMMON_FLAGS -O3 -ffinite-math-only -Qstrip_debug -Qstrip_reflect)

    if (GGML_DX12_XBOX)
        # The Scarlett dxc adds a "precompilation" pass after the normal DXIL
        # compile that tries to pre-build a Scarlett PSO from the container.
        # That PSO build requires a root signature embedded in the DXIL
        # (e.g. via [RootSignature("...")] on the entry point or
        # -rootsig-define). ggml-dx12 creates a single shared common_root_sig
        # at runtime instead, so the precompile aborts with:
        #   "; Precompilation failed - Could not find root signature in dxil container"
        # which the build treats as a fatal error.
        #
        # -noprecompile disables only the Scarlett PSO precompile pass; the
        # DXIL container is still produced and is fully usable at runtime.
        list(APPEND DXC_COMMON_FLAGS -noprecompile)
    endif()

    # Shaders that must NOT use -ffinite-math-only (NaN/Inf sensitivity)
    set(DX12_SHADERS_NO_FINITE_MATH
        flash_attn
        flash_attn_64
        flash_attn_128
        flash_attn_gqa
        flash_attn_reduce
    )

    # Compile each shader for wave sizes 16, 32, 64
    # At runtime, the device selects the matching set based on GPU wave size
    set(DX12_WAVE_SIZES 16 32 64)

    # Shader files that are #included by other shaders (transitive deps).
    # Add to DEPENDS so changes trigger rebuilds of dependent shaders.
    set(SHADER_INCLUDE_DEPS
        ${SHADER_INPUT_DIR}/ggml_common.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_id_quant.hlsli
        ${SHADER_INPUT_DIR}/rope_yarn.hlsli
        ${SHADER_INPUT_DIR}/flash_attn.hlsl
        ${SHADER_INPUT_DIR}/mul_mat_vec_q4k_dp4a.hlsl
        ${SHADER_INPUT_DIR}/mul_mat_vec_q5k_dp4a.hlsl
    )

    foreach(WSIZE ${DX12_WAVE_SIZES})
        foreach(shader_name ${DX12_SHADERS})
            set(shader_src "${SHADER_INPUT_DIR}/${shader_name}.hlsl")
            set(shader_hdr "${SHADER_OUTPUT_DIR}/${shader_name}_w${WSIZE}_dxil.h")

            # Check if this shader needs safe math
            set(SHADER_DXC_FLAGS ${DXC_COMMON_FLAGS})
            if(shader_name IN_LIST DX12_SHADERS_NO_FINITE_MATH)
                list(REMOVE_ITEM SHADER_DXC_FLAGS -ffinite-math-only)
            endif()

            add_custom_command(
                OUTPUT ${shader_hdr}
                COMMAND ${DXC_EXECUTABLE}
                    -T cs_6_6
                    -E main
                    -Fh ${shader_hdr}
                    -Vn g_${shader_name}_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    ${SHADER_DXC_FLAGS}
                    -D WAVE_SIZE=${WSIZE}
                    ${shader_src}
                DEPENDS ${shader_src} ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader: ${shader_name} (wave=${WSIZE})"
            )
            list(APPEND DX12_SHADER_HEADERS ${shader_hdr})
        endforeach()

        foreach(shader_name ${DX12_SHADERS_SM64})
            set(shader_src "${SHADER_INPUT_DIR}/${shader_name}.hlsl")
            set(shader_hdr "${SHADER_OUTPUT_DIR}/${shader_name}_w${WSIZE}_dxil.h")

            set(SHADER_DXC_FLAGS ${DXC_COMMON_FLAGS})
            if(shader_name IN_LIST DX12_SHADERS_NO_FINITE_MATH)
                list(REMOVE_ITEM SHADER_DXC_FLAGS -ffinite-math-only)
            endif()

            add_custom_command(
                OUTPUT ${shader_hdr}
                COMMAND ${DXC_EXECUTABLE}
                    -T cs_6_6
                    -E main
                    -Fh ${shader_hdr}
                    -Vn g_${shader_name}_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    ${SHADER_DXC_FLAGS}
                    -D WAVE_SIZE=${WSIZE}
                    ${shader_src}
                DEPENDS ${shader_src} ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.6): ${shader_name} (wave=${WSIZE})"
            )
            list(APPEND DX12_SHADER_HEADERS ${shader_hdr})
        endforeach()
    endforeach()

    # Generate a master include header
    set(SHADER_MASTER_HEADER "${SHADER_OUTPUT_DIR}/ggml_dx12_shaders.h")
    set(SHADER_INCLUDES "")
    foreach(WSIZE ${DX12_WAVE_SIZES})
        foreach(shader_name ${DX12_SHADERS})
            string(APPEND SHADER_INCLUDES "#include \"${shader_name}_w${WSIZE}_dxil.h\"\n")
        endforeach()
        foreach(shader_name ${DX12_SHADERS_SM64})
            string(APPEND SHADER_INCLUDES "#include \"${shader_name}_w${WSIZE}_dxil.h\"\n")
        endforeach()
    endforeach()
    file(WRITE "${SHADER_MASTER_HEADER}" "// Auto-generated shader includes\n#pragma once\n${SHADER_INCLUDES}")

    target_sources(ggml-dx12 PRIVATE ${DX12_SHADER_HEADERS})
    target_include_directories(ggml-dx12 PRIVATE ${SHADER_OUTPUT_DIR})
    target_compile_definitions(ggml-dx12 PRIVATE GGML_DX12_SHADERS_COMPILED)
else()
    message(WARNING "ggml-dx12: DXC not found - HLSL shaders will not be compiled, compute operations will be unavailable")
endif()
