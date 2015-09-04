#ifndef SAMU_HPP
#define SAMU_HPP

/**
 * @brief SAMU - the potential ancestor of developmental robotics chatter bots
 *
 * @file samu.hpp
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
#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>

#include "nlp.hpp"
#include "vi.hpp"

class Samu
{
public:

  Samu()
  {
    cv_.notify_one();
  }

  ~Samu()
  {
    run_ = false;
    terminal_thread_.join();
  }

  bool run ( void )
  {
    return run_;
  }

  void FamilyCaregiverShell ( void );
  void terminal ( void )
  {
    std::unique_lock<std::mutex> lk ( mutex_ );
    cv_.wait ( lk );

    FamilyCaregiverShell();
  }

  void operator<< ( std::string sentence )
  {
    vi << nlp.sentence2triplets ( sentence.c_str() );
  }

  std::string Caregiver()
  {
    if ( caregiver_name_.size() > 0 )
      return caregiver_name_[caregiver_idx_];
    else
      return "Undefined";
  }

  void NextCaregiver()
  {
    caregiver_idx_ = ( caregiver_idx_ + 1 ) % caregiver_name_.size();
  }

  double reward ( void )
  {
    return vi.reward();
  }

private:

  bool run_ {true};
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread terminal_thread_ {&Samu::terminal, this};

  NLP nlp;
  VisualImagery vi;

  int caregiver_idx_ {0};
  std::vector<std::string> caregiver_name_ {"Norbi", "Nandi", "Matyi", "Greta"};
};

#endif
