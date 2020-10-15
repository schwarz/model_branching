with import <nixpkgs> {};

let
  pythonEnv = python37.withPackages (
    ps: [
      ps.beautifulsoup4
      ps.black
      ps.numpy
      ps.pandas
      ps.pip
      ps.pyarrow
    ]
  );
in mkShell {
  buildInputs = [
    pythonEnv
  ];

}
