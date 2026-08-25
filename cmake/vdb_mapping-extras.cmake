# The package ships FindOpenVDB/FindZSTD modules because those dependencies do
# not provide portable config packages on every supported platform. Make the
# modules visible before ament's exported dependency hooks run.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

# ament_export_dependencies cannot express PCL components and would make
# every downstream consumer discover/link the full CUDA/VTK/Qt PCL surface.
# The public headers use only the common and PCD-I/O portions.
find_package(PCL REQUIRED COMPONENTS common io)
