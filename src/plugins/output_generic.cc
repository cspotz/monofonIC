/*
 
 output_generic.cc - This file is part of MUSIC2 - GPL
 Copyright (C) 2010-19  Oliver Hahn
 
 */

#ifdef USE_HDF5

#include <unistd.h> // for unlink

#include "HDF_IO.hh"
#include <logger.hh>
#include <output_plugin.hh>

class generic_output_plugin : public output_plugin
{
private:
	std::string get_field_name( const cosmo_species &s, const fluid_component &c );

public:
	//! constructor
	explicit generic_output_plugin(ConfigFile &cf )
	: output_plugin(cf, "Generic HDF5")
	{
		real_t astart  = 1.0/(1.0+cf_.GetValue<double>("setup", "zstart"));
		real_t boxsize = cf_.GetValue<double>("setup", "BoxLength");

	#if defined(USE_MPI)
        if( CONFIG::MPI_task_rank == 0 )
            unlink(fname_.c_str());
        MPI_Barrier( MPI_COMM_WORLD );
	#else
        unlink(fname_.c_str());
	#endif

		HDFCreateFile( fname_ );
		HDFCreateGroup( fname_, "Header" );
		HDFWriteGroupAttribute<double>( fname_, "Header", "Boxsize", boxsize );
		HDFWriteGroupAttribute<double>( fname_, "Header", "astart", astart );
	}

    bool write_species_as_grid( const cosmo_species & ){ return true; }
	real_t position_unit() const { return 1.0; }
	real_t velocity_unit() const { return 1.0; }
	void write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c );
};


std::string generic_output_plugin::get_field_name( const cosmo_species &s, const fluid_component &c )
{
	std::string field_name;
	switch( s ){
		case cosmo_species::dm: 
			field_name += "DM"; break;
		case cosmo_species::baryon: 
			field_name += "BA"; break;
		case cosmo_species::neutrino: 
			field_name += "NU"; break;
		default: break;
	}
	field_name += "_";
	switch( c ){
		case fluid_component::density:
			field_name += "delta"; break;
		case fluid_component::vx:
			field_name += "vx"; break;
		case fluid_component::vy:
			field_name += "vy"; break;
		case fluid_component::vz:
			field_name += "vz"; break;
		case fluid_component::dx:
			field_name += "dx"; break;
		case fluid_component::dy:
			field_name += "dy"; break;
		case fluid_component::dz:
			field_name += "dz"; break;
		default: break;
	}
	return field_name;
}

void generic_output_plugin::write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c ) 
{
	std::string field_name = this->get_field_name( s, c );
	g.Write_to_HDF5(fname_, field_name);
	csoca::ilog << interface_name_ << " : Wrote field \'" << field_name << "\' to file \'" << fname_ << "\'" << std::endl;
}

namespace
{
   output_plugin_creator_concrete<generic_output_plugin> creator1("generic"); 
} // namespace

#endif