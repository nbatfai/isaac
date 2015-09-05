/**
 * @brief ISAAC - deep Q learning with neural networks for predicting the next sentence of a conversation
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

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

void Samu::FamilyCaregiverShell ( void )
{
  std::string cmd_prefix = "___";

  fd_set rfds;
  struct timeval tmo;

  FD_ZERO ( &rfds );

  std::cout << Caregiver() << "@Caregiver> " << std::flush;

  int sleep {0};
  
  if(sleep_)
    sleep = sleep_after_ + 1;
  
  for ( ; run_ ; )
    {

      FD_SET ( 0, &rfds );

      tmo.tv_sec = 1;
      tmo.tv_usec = 0;

      int s = select ( 1, &rfds, NULL, NULL, &tmo );

      if ( s == -1 )
        {
          perror ( "select" );
        }
      else if ( s )
        {
          std::string line;
          std::getline ( std::cin,  line );

	  if(sleep_)
	    std::cout << "Isaac is awake now." << std::endl;
	    
          sleep_ = false;
	  sleep = 0;

          if ( !line.compare ( 0, cmd_prefix.length(), cmd_prefix ) )
            {
              if ( line == cmd_prefix )
                NextCaregiver();
            }
          else
            {
              *this << line;
            }

        }
      else
        {
	  if ( ++sleep > sleep_after_ )
	  {
	    if(!sleep_)
	      std::cout << "Isaac went to sleep." << std::endl;
            sleep_ = true;
	  }
	  else
	  {
	      std::cout << sleep << " " << std::endl;	    
	  }
	    
        }

      //std::cout << Caregiver() << "@Caregiver> " << std::endl;
    }

  run_ = false;
}
