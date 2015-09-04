/**
 * @brief SAMU - the potential ancestor of developmental robotics chatter bots
 *
 * @file samu.cpp
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
#include <string>
#include <sstream>
#include "samu.hpp"

void Samu::FamilyCaregiverShell(void)
{
  std::string cmd_prefix = "___";

  std::cout << Caregiver() << "@Caregiver> " << std::flush;

  for ( std::string line; std::getline ( std::cin,  line ); )
    {

      if ( !line.compare ( 0, cmd_prefix.length(), cmd_prefix ) )
        {
          if ( line == cmd_prefix )
            NextCaregiver();
        }
      else
        {
          *this << line;
        }

      std::cout << Caregiver() << "@Caregiver> " << std::flush;
    }
    
    run_ = false;    
}
