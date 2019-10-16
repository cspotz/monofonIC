/*
 
 output.cc - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010  Oliver Hahn
 
*/

#include "output_plugin.hh"


std::map< std::string, output_plugin_creator *>& get_output_plugin_map()
{
	static std::map< std::string, output_plugin_creator* > output_plugin_map;
	return output_plugin_map;
}

void print_output_plugins()
{
	std::map< std::string, output_plugin_creator *>& m = get_output_plugin_map();
	
	std::map< std::string, output_plugin_creator *>::iterator it;
	it = m.begin();
	csoca::ilog << "Available output plug-ins:\n";
	while( it!=m.end() )
	{
		if( it->second )
			csoca::ilog << "\t\'" << it->first << "\'\n";
		++it;
	}
}

std::unique_ptr<output_plugin> select_output_plugin( ConfigFile& cf )
{
	std::string formatname = cf.GetValue<std::string>( "output", "format" );
	
	output_plugin_creator *the_output_plugin_creator 
	= get_output_plugin_map()[ formatname ];
	
	if( !the_output_plugin_creator )
	{	
		csoca::elog << "Error: output plug-in \'" << formatname << "\' not found." << std::endl;
		print_output_plugins();
		throw std::runtime_error("Unknown output plug-in");
		
	}else{
		csoca::ilog << "-------------------------------------------------------------------------------" << std::endl;
        csoca::ilog << std::setw(32) << std::left << "Output plugin" << " : " << formatname << std::endl;
	}
	
	return std::move(the_output_plugin_creator->create( cf ));
}



