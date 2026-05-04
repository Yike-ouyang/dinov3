{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  hatchling,
  numpy,
  pynng,
  requests,
  pytest,
  loguru,
  matplotlib,
}:
buildPythonPackage {
  pname = "nahual";
  version = "0.0.9";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "afermg";
    repo = "nahual";
    rev = "dd58a2506fe5172fe39deaca260b4fe7ee1ef313";
    sha256 = "sha256-02je/noKv3/iD7TjR2+4mOrGbX0wEZ+XQMDM2vlLZ9U=";
  };

  build-system = [
    hatchling
  ];

  dependencies = [
    numpy
    pynng
    requests
    pytest
    loguru
    matplotlib
  ];

  pythonImportsCheck = [
    "nahual"
  ];

  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  meta = {
    description = "Deploy and access image and data processing models across processes.";
    homepage = "https://github.com/afermg/nahual";
    license = lib.licenses.mit;
  };
}
