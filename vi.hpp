#ifndef VI_HPP
#define VI_HPP

/**
 * @brief ISAAC - deep Q learning with neural networks for predicting the next sentence of a conversation
 *
 * @file vi.hpp
 * @author  Norbert Bátfai <nbatfai@gmail.com>
 * @version 0.0.1
 *
 * @section LICENSE
 *
 * Copyright (C) 2015 Norbert Bátfai, batfai.norbert@inf.unideb.hu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @section DESCRIPTION
 * SAMU
 *
 * The main purpose of this project is to allow the evaluation and
 * verification of the results of the paper entitled "A disembodied
 * developmental robotic agent called Samu Bátfai". It is our hope
 * that Samu will be the ancestor of developmental robotics chatter
 * bots that will be able to chat in natural language like humans do.
 *
 */

#include <iostream>
#include <queue>
#include <cstdio>
#include <cstring>

#include "nlp.hpp"
#include "ql.hpp"

#include <pngwriter.h>
#include <chrono>
#include <boost/date_time/posix_time/posix_time.hpp>

class VisualImagery
{
public:
  VisualImagery()
  {}

  ~VisualImagery()
  {}

  void operator<< ( std::vector<SPOTriplet> triplets )
  {

    if ( !triplets.size() )
      return;

    for ( auto triplet : triplets )
      {
        if ( program.size() >= stmt_max )
          program.pop();

        program.push ( triplet );
      }

    boost::posix_time::ptime now = boost::posix_time::second_clock::universal_time();
    std::string image_file = "samu_vi_"+boost::posix_time::to_simple_string ( now ) +".png";

    char * image_file_p = strdup ( image_file.c_str() );
    pngwriter image ( 256, 256, 65535, image_file_p );
    free ( image_file_p );

    char stmt_buffer[1024];
    char *stmt_buffer_p = stmt_buffer;

    std::queue<SPOTriplet> run = program;

#ifndef Q_LOOKUP_TABLE

    std::string prg;
    stmt_counter = 0;
    while ( !run.empty() )
      {
        auto triplet = run.front();

        prg += triplet.s.c_str();
        prg += triplet.p.c_str();
        prg += triplet.o.c_str();
	
        std::snprintf ( stmt_buffer, 1024, "%s.%s(%s);", triplet.s.c_str(), triplet.p.c_str(), triplet.o.c_str() );

        char font[] = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf";
        char *font_p = font;

        image.plot_text_utf8 ( font_p,
                               11,
                               5,
                               256- ( ++stmt_counter ) *28,
                               0.0,
                               stmt_buffer_p, 0, 0, 0 );
        run.pop();
      }

    double *img_input = new double[256*256];

    for ( int i {0}; i<256; ++i )
      for ( int j {0}; j<256; ++j )
        {
          img_input[i*256+j] = image.dread ( i, j );
        }
#else
    std::string prg;
    while ( !run.empty() )
      {
        auto triplet = run.front();

        prg += triplet.s.c_str();
        prg += triplet.p.c_str();
        prg += triplet.o.c_str();

        run.pop();
      }
#endif

    auto start = std::chrono::high_resolution_clock::now();

    std::cerr << "QL start... ";

#ifndef Q_LOOKUP_TABLE

    SPOTriplet response = ql ( triplets[0], prg, img_input );

    std::cout << std::endl
              << "Isaac@AI"
#ifdef QNN_DEBUG	      
              << "."
              << ql.get_action_count()
              << "."
              << ql.get_action_relevance()
              << "%"
#endif	      
	      <<"> "
              << response << std::endl;

#else

    std::cerr << ql ( triplets[0], prg ) << std::endl;

#endif


    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds> (
                std::chrono::high_resolution_clock::now() - start ).count()
              << " ms " <<  std::endl;


#ifndef Q_LOOKUP_TABLE
    delete[] img_input;
    image.close();
#endif
  }

  double reward ( void )
  {
    return ql.reward();
  }

  void save(std::string &fname)
  {
    ql.save(fname);
  }  
  
  void load(std::fstream & file)
  {    
    ql.load(file);
  }    

  void t(void)
  {
    while ( !program.empty() )
      {
        program.pop();
      }
  }
  
private:

  QL ql;
  std::queue<SPOTriplet> program;
  int stmt_counter {0};
  static const int stmt_max = 7;

};

#endif
