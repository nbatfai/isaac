#ifndef NLP_HPP
#define NLP_HPP

/**
 * @brief SAMU - the potential ancestor of developmental robotics chatter bots
 *
 * @file nlp.hpp
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

#include <locale.h>
#include <stdio.h>
#include "link-grammar/link-includes.h"
#include <string>
#include <vector>
#include <iostream>

class SPOTriplet
{

public:

  std::string s;
  std::string p;
  std::string o;

  SPOTriplet ()
  {}

  SPOTriplet ( std::string &s, std::string &p, std::string &o ) :s ( s ),p ( p ), o ( o )
  {}

  friend std::ostream & operator<< ( std::ostream & os, const SPOTriplet & t )
  {
    std::cout << t.s << " " << t.p << " "<< t.o;

    return os;
  }

  const bool operator== ( const SPOTriplet &other ) const
  {
    if ( s == other.s &&
         p == other.p &&
         o == other.o )
      return true;
    else
      return false;
  }

  const bool operator< ( const SPOTriplet &other ) const
  {
    return s+p+o < other.s+other.p+other.o;
  }

  const double cmp ( const SPOTriplet &other ) const
  {
    if ( *this == other )
      return 1.0;
    else if ( ( s == other.s &&
                p == other.p ) ||
              ( s == other.s &&
                o == other.o ) ||
              ( o == other.o &&
                p == other.p ) )
      return 2.0/3.0;
    else     if ( s == other.s ||
                  p == other.p ||
                  o == other.o )
      return 1.0/3.0;
    else
      return 0.0;
  }

  void cut ( )
  {
    cut ( s );
    cut ( p );
    cut ( o );
  }

  void cut ( std::string &s )
  {
    std::size_t found = s.find ( '.' );
    if ( found != std::string::npos )
      s.erase ( found );

    found = s.find ( '[' );
    if ( found != std::string::npos )
      s.erase ( found );
  }
  
};

typedef std::vector<SPOTriplet> SPOTriplets;

class NLP
{
public:
  NLP()
  {
    setlocale ( LC_ALL, "en_US.UTF8" );
    parse_opts_ = parse_options_create();
    dict_ = dictionary_create_lang ( "en" );

    if ( !dict_ )
      {
        std::cerr << "Link grammar dictionary cannot be found." << std::endl ;
      }

    std::cout << linkgrammar_get_version() << std::endl ;

  }

  ~NLP()
  {
    dictionary_delete ( dict_ );
    parse_options_delete ( parse_opts_ );
  }

  SPOTriplets sentence2triplets ( const char* );

private:

  Dictionary    dict_;
  Parse_Options parse_opts_;

};

#endif
