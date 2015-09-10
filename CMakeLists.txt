 #
 # @brief ISAAC - deep Q learning with neural networks for predicting the next sentence of a conversation
 #
 # @file CMakeLists.txt
 # @author  Norbert Bátfai <nbatfai@gmail.com>
 # @version 0.0.1
 #
 # @section LICENSE
 #
 # Copyright (C) 2015 Norbert Bátfai, batfai.norbert@inf.unideb.hu
 #
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 #
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program.  If not, see <http://www.gnu.org/licenses/>.
 #
 # @section DESCRIPTION
 #
 # ISAAC
 #
 # "The son of Samu is Isaac." The project called Isaac is a case study 
 # of using deep Q learning with neural networks for predicting the next 
 # sentence of a conversation.
 # 
 # SAMU
 #
 # The main purpose of this project is to allow the evaluation and 
 # verification of the results of the paper entitled "A disembodied 
 # developmental robotic agent called Samu Bátfai". It is our hope 
 # that Samu will be the ancestor of developmental robotics chatter 
 # bots that will be able to chat in natural language like humans do.
 #
 
cmake_minimum_required(VERSION 2.8.5)
project(SAMU)

#set(CUDA_LAYERS ON)
set(CUDA_LAYERS OFF)

# uncomment to use Q lookup table
#add_definitions(-DQ_LOOKUP_TABLE)

add_definitions(-DQNN_DEBUG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3")

if(${CUDA_LAYERS})

  add_definitions(-DCUDA_PRCPS)
  find_package(CUDA REQUIRED)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -std=c++11 
  -gencode arch=compute_30,code=sm_30)

endif()
  
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package( Boost COMPONENTS date_time REQUIRED )  

include_directories( ${Boost_INCLUDE_DIRS} ) 
link_directories( ${Boost_LIBRARY_DIRS} )  

include(FindPNGwriter.cmake)

if(PNGwriter_FOUND)
  include_directories(SYSTEM ${PNGwriter_INCLUDE_DIRS})
  add_definitions(${PNGwriter_DEFINITIONS})
  set(LIBS ${LIBS} ${PNGwriter_LIBRARIES})
endif(PNGwriter_FOUND)

include(FindLinkGrammar.cmake)

INCLUDE_DIRECTORIES(${LINK_GRAMMAR_INCLUDE_DIRS})

if (CUDA_FOUND)

  cuda_compile(CUDASRCS qlc.cu)
  cuda_add_executable(samu ${CUDASRCS} nlp.hpp nlp.cpp vi.hpp qlc.h ql.hpp samu.hpp samu.cpp main.cpp  )

else()

  add_executable(samu nlp.hpp nlp.cpp vi.hpp ql.hpp samu.hpp samu.cpp main.cpp  )

endif()

target_link_libraries(samu ${LINK_GRAMMAR_LIBRARIES} ${PNGwriter_LIBRARIES} ${Boost_LIBRARIES})

install(TARGETS samu DESTINATION bin)
