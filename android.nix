{ stdenvNoCC, lib, fetchFromGitHub, autoreconfHook, libtool, ndk, cmake, pkg-config, git, bladerf-src }:
rec {
  libusb = stdenvNoCC.mkDerivation rec {
    pname = "libusb";
    version = "1.0.27";

    src = fetchFromGitHub {
      owner = "libusb";
      repo = "libusba";
      rev = "v${version}";
      hash = "sha256-J6LRlDp+JOCTAooY5yWNENSwslp3HZ8/MRwKQfxEMaM=";
    };

    # NDK contains all build inputs
    nativeBuildInputs = [];

    NDK = "${ndk}";

    buildPhase = ''
      cd android/jni
      export NDK="${NDK}"
      echo "NDK: $NDK"
      $NDK/ndk-build
    '';

    installPhase = ''
      mkdir -p $out/lib
      mkdir -p $out/include/libusb-1.0

      # mkdir -p $out/lib/arm64-v8a
      # mkdir -p $out/lib/armeabi-v7a
      # mkdir -p $out/lib/x86
      # mkdir -p $out/lib/x86_64

      # # Copy the built library and headers to $out

      # cp -v ../libs/arm64-v8a/libusb1.0.so $out/lib/arm64-v8a
      # cp -v ../libs/armeabi-v7a/libusb1.0.so $out/lib/armeabi-v7a
      # cp -v ../libs/x86/libusb1.0.so $out/lib/x86
      # cp -v ../libs/x86_64/libusb1.0.so $out/lib/x86_64

      # Copy just arm v8 for now. Looks like cmake can only handle one so at a time
      cp -v ../libs/arm64-v8a/libusb1.0.so $out/lib/libusb-1.0.so

      cp -v ../../libusb/libusb.h $out/include/libusb-1.0/
      cp -v ../../libusb/version.h $out/include/libusb-1.0/
    '';
  };
  libbladerf = stdenvNoCC.mkDerivation rec {
    pname = "libbladeRF";
    version = "master";

    src = bladerf-src;

    nativeBuildInputs = [ cmake pkg-config git ];
    buildInputs = [ libusb ];

    NDK_TOOLCHAIN = "${ndk}/toolchains/llvm/prebuilt/linux-x86_64/";

    cmakeFlags = [
      "-DCMAKE_TOOLCHAIN_FILE=${ndk}/build/cmake/android.toolchain.cmake"
      "-DANDROID_NDK=${ndk}"
      "-DANDROID_ABI=arm64-v8a"
      "-DANDROID_PLATFORM=android-21"
      "-DCMAKE_SYSTEM_NAME=Android"
      "-DCMAKE_VERBOSE_MAKEFILE=ON"
      "-DENABLE_BACKEND_LIBUSB=ON"
      "-DCMAKE_FIND_ROOT_PATH=${libusb};${ndk}/sysroot"
      "-DCMAKE_INSTALL_PREFIX=$out"
      "-DBUILD_DOCUMENTATION=OFF"
      "-DVERSION_INFO_OVERRIDE=foxhunter-${builtins.substring 0 7 src.rev}"
    ];

    preConfigure = ''
      echo "NDK_TOOLCHAIN: $NDK_TOOLCHAIN"
      echo "LIBUSB_PATH: ${libusb}"

      export CC="${NDK_TOOLCHAIN}/bin/aarch64-linux-android21-clang"
      export CXX="${NDK_TOOLCHAIN}/bin/aarch64-linux-android21-clang++"
      export AR="${NDK_TOOLCHAIN}/bin/aarch64-linux-android-ar"
      export AS="${NDK_TOOLCHAIN}/bin/aarch64-linux-android-as"
      export LD="${NDK_TOOLCHAIN}/bin/aarch64-linux-android-ld"
      export RANLIB="${NDK_TOOLCHAIN}/bin/aarch64-linux-android-ranlib"
      export STRIP="${NDK_TOOLCHAIN}/bin/aarch64-linux-android-strip"
    '';

    configurePhase = ''
      set -x
      cmake -B build -S . ${lib.escapeShellArgs cmakeFlags}
    '';

    buildPhase = ''
      set -x
      cmake --build build
    '';

    installPhase = ''
      set -x
      cmake --install build --prefix $out
    '';
  };
}