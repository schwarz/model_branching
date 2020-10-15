with import <nixpkgs> {};

let
  pythonEnv = python37.withPackages (
    ps: [
    ps.numpy
    ps.scipy
    ps.matplotlib
    ps.pandas
    ps.jupyter
    ps.pyarrow
    ps.pytorch
    ps.black # dev
    ps.pip # dev
    ]
  );
in mkShell {
  buildInputs = [
    pythonEnv
    texlive.combined.scheme-full # latex plotting
  ];

}
