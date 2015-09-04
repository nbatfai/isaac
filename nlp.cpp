/**
 * @brief SAMU - the potential ancestor of developmental robotics chatter bots
 *
 * @file nlp.cpp
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

#include "nlp.hpp"

SPOTriplets NLP::sentence2triplets ( const char* sentence )
{
  SPOTriplets triplets;

  Sentence sent = sentence_create ( sentence, dict_ );
  sentence_split ( sent, parse_opts_ );
  int num_linkages = sentence_parse ( sent, parse_opts_ );

  SPOTriplet triplet;
  std::string alter_p;

  bool ready = false;

  for ( int l {0}; l< num_linkages && !ready; ++l )
    {
      Linkage linkage = linkage_create ( l, sent, parse_opts_ );

      std::vector<std::string> words;

      for ( int k=0; k<linkage_get_num_links ( linkage ); ++k )
        {

          const char *p = linkage_get_word ( linkage, k );
          if ( p )
            words.push_back ( p );
          else
            words.push_back ( "null" );

        }

      for ( int k {0}; k<linkage_get_num_links ( linkage ) && !ready; ++k )
        {

          const char* c = linkage_get_link_label ( linkage, k );

          if ( *c == 'S' )
            {
              triplet.p = linkage_get_word ( linkage, k );
              alter_p = words[linkage_get_link_rword ( linkage, k )];
              triplet.s = words[linkage_get_link_lword ( linkage, k )];
            }

          if ( *c == 'O' )
            {
              triplet.o = words[linkage_get_link_rword ( linkage, k )];

              if ( triplet.p == words[linkage_get_link_lword ( linkage, k )] )
                {

                  triplet.cut ( );

                  triplets.push_back ( triplet );

                  ready = true;
                  break;
                }
              else if ( alter_p == words[linkage_get_link_lword ( linkage, k )] )
                {
                  triplet.p = alter_p;

                  triplet.cut ( );

                  triplets.push_back ( triplet );

                  ready = true;
                  break;

                }
            }
        }

      linkage_delete ( linkage );
    }

  sentence_delete ( sent );

  return triplets;
}
