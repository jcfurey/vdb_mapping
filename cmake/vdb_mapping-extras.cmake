# The package ships FindOpenVDB/FindZSTD modules because those dependencies do
# not provide portable config packages on every supported platform. Make the
# modules visible before ament's exported dependency hooks run.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
