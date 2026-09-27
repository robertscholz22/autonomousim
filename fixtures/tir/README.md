# Reference tyre property files

`MagicFormula52_Parameters.tir`, `MagicFormula61_Parameters.tir` and `MagicFormula62_Parameters.tir`
are the sample files of
[MFeval.jl](https://github.com/matheusft/MFeval_julia) ("MFeval.jl - High-Performance Pacejka
Magic Formula Tyre Model Implementation", MIT license), itself a port of Marco Furlan's MATLAB
[mfeval](https://www.mathworks.com/matlabcentral/fileexchange/63618-mfeval). They exercise the
MF 5.2, MF 6.1 and MF 6.2 equations (camber up to ±55°, pressure, turn slip) in
`crates/vehicles/tests/tire_fixtures.rs`.
