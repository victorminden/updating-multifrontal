# add_executable(TEST test.cpp)
# target_link_libraries(TEST ${MATH_LIBS})
include_directories(${HIF_UPDATE_SPARSE_SOURCE_DIR}/test_include/DE)
link_directories(${HIF_UPDATE_SPARSE_SOURCE_DIR}/test_src/DE)

add_executable(testMF2D testMF2D.cpp)
target_link_libraries(testMF2D ${MATH_LIBS})

add_executable(testMF3D testMF3D.cpp)
target_link_libraries(testMF3D ${MATH_LIBS})


if(GPROF)
  SET_TARGET_PROPERTIES(testMF2D PROPERTIES LINK_FLAGS -pg)
  SET_TARGET_PROPERTIES(testMF3D PROPERTIES LINK_FLAGS -pg)
endif()