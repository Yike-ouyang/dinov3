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
  version = "0.0.8";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "afermg";
    repo = "nahual";
    rev = "87c4cecd5782acc42e4878ca6a439ac227663b69";
    sha256 = "sha256-oVoffChdzb2S3gTfCnhBKRUaGmI0auFJwo54+UVF4Sw=";
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
