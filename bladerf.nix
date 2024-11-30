{ lib, isDarwin, bladerf-src, fetchurl, fetchFromGitHub, fetchpatch, libbladeRF, symlinkJoin }:
rec {
  xa4-bitstream = fetchurl {
    # See: https://www.nuand.com/fpga_images/
    url = "https://www.nuand.com/fpga/v0.15.3/hostedxA4.rbf";
    sha256 = "sha256-6qQVZQtrAOdfHijCqGDY+QV3sfRkiy97iKZXRfRkpts=";
  };
  fx3-firmware = fetchurl {
    # See: https://www.nuand.com/fx3_images/
    url = "https://www.nuand.com/fx3/bladeRF_fw_v2.4.0.img";
    sha256 = "sha256-Zw0cp6ocYAfrCZADUcOqmX5L4xbbwKL8FTKpCNAswKk=";
  };
  libbladerf = libbladeRF.overrideAttrs (oldAttrs: rec {
    version = "master";
    src = bladerf-src;
    # bladeRF-fsk (cli) support for BladeRf 2.0 micro
    patches = oldAttrs.patches or [] ++ [
      (fetchpatch {
        url = "https://github.com/Nuand/bladeRF/commit/2db141bf6225abd3cc51e64da14461739bab35dc.patch";
        sha256 = "sha256-UHRw7HkjYFwRbGQki5l5vSaxLbhYFrWsN6ZEYSjYB2s=";
      })
    ];
    cmakeFlags = oldAttrs.cmakeFlags or [] ++ [
      # FIXME: HACK
      "-DVERSION_INFO_OVERRIDE=foxhunter-${builtins.substring 0 7 /*src.rev*/ "DEADBEEF"}"
    ] ++ lib.optionals isDarwin [
      "-DCMAKE_C_FLAGS=-Wno-error=format"
    ];
  }); 
}
