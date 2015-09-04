 #
 # @brief SAMU - the potential ancestor of developmental robotics chatter bots
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

#add_definitions(-DQ_LOOKUP_TABLE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

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

add_executable(samu nlp.hpp nlp.cpp vi.hpp ql.hpp samu.hpp samu.cpp main.cpp)

target_link_libraries(samu ${LINK_GRAMMAR_LIBRARIES} ${PNGwriter_LIBRARIES} ${Boost_LIBRARIES})

install(TARGETS samu DESTINATION bin)
