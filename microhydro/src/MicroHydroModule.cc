// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MicroHydroModule.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Bench MicroHydro.                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define INCLUDE_EDGES 0

#include <arcane/utils/ApplicationInfo.h>
#include <arcane/utils/CommandLineArguments.h>
#include <arcane/utils/MemoryUtils.h>

#include <arcane/core/IMesh.h>
#include <arcane/core/IParallelMng.h>
#include <arcane/core/ISimpleTableComparator.h>
#include <arcane/core/ISimpleTableOutput.h>
#include <arcane/core/ISubDomain.h>
#include <arcane/core/ITimeLoopMng.h>
#include <arcane/core/ITimeStats.h>
#include <arcane/core/ItemEnumerator.h>
#include <arcane/core/ItemPrinter.h>
#include <arcane/core/MathUtils.h>
#include <arcane/core/ModuleFactory.h>
#include <arcane/core/UnstructuredMeshConnectivity.h>
#include <arcane/core/VariableTypes.h>

#include <arcane/accelerator/Reduce.h>
#include <arcane/accelerator/RunCommandEnumerate.h>
#include <arcane/accelerator/VariableViews.h>
#include <arcane/accelerator/core/IAcceleratorMng.h>
#include <arcane/accelerator/core/Runner.h>

#include "arcane/IMesh.h"
#include "arcane/IParallelMng.h"
#include "arcane/ISubDomain.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/ITimeStats.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/ItemPrinter.h"
#include "arcane/MathUtils.h"
#include "arcane/ModuleFactory.h"
#include "arcane/VariableTypes.h"

#include "arcane/mesh/ItemFamily.h"

#include <arcane/Connectivity.h>

#include "MicroHydroTypes.h"
#include "MicroHydro_axl.h"

#include "connectivix/ConnectivityMatrix.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace MicroHydro {

namespace ax = Arcane::Accelerator;
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module MicroHydro.
 *
 * Ce module implémente une hydrodynamique simple tri-dimensionnel,
 * parallèle, avec une pseudo-viscosité aux mailles.
 */
class MicroHydroModule : public ArcaneMicroHydroObject {
public:
  struct BoundaryCondition {
    NodeGroup nodes;
    NodeVectorView view;
    Real value;
    MicroHydroTypes::eBoundaryCondition type;
  };

  // Note: il faut mettre ce champs statique si on veut que sa valeur
  // soit correcte lors de la capture avec CUDA (sinon on passe par this et
  // cela provoque une erreur mémoire)
  static const Integer MAX_NODE_CELL = 8;

public:
  //! Constructeur
  explicit MicroHydroModule(const ModuleBuildInfo &mb);

public:
  VersionInfo versionInfo() const override {
    return VersionInfo(2, 0, 1);
  }

public:
  void hydroBuild() override;
  void hydroStartInit() override;
  void hydroInit() override;
  void hydroExit() override;
  void hydroOnMeshChanged() override;
  void doOneIteration() override;

public:
  void computeForces();
  void computeVelocity();
  void computeViscosityWork();
  void applyBoundaryCondition();
  void moveNodes();
  void computeGeometricValues();
  void updateDensity();
  void applyEquationOfState();
  void computeDeltaT();

  void _computeNodeIndexInCells();

private:
  ITimeStats *m_time_stats = nullptr;
  Timer m_elapsed_timer;

  //! Indice de chaque noeud dans la maille
  UniqueArray<Int16> m_node_index_in_cells;

  Runner m_runner;
  RunQueue m_default_queue;

  UnstructuredMeshConnectivityView m_connectivity_view;
  UniqueArray<BoundaryCondition> m_boundary_conditions;

  Connectivix::ConnectivityMatrix<Node, Edge> *m_edgesOfNode;
  Connectivix::ConnectivityMatrix<Node, Face> *m_facesOfNode;
  Connectivix::ConnectivityMatrix<Face, Cell> *m_cellsOfFace;

  Connectivix::ConnectivityMatrix<Edge, Node> *m_nodesOfEdge;
  Connectivix::ConnectivityMatrix<Face, Node> *m_nodesOfFace;
  Connectivix::ConnectivityMatrix<Cell, Face> *m_facesOfCell;
  Connectivix::ConnectivityMatrix<Face, Edge> *m_edgesOfFace;

  Connectivix::ConnectivityMatrix<Cell, Node> *m_nodesOfCell;
  Connectivix::ConnectivityMatrix<Node, Cell> *m_cellsOfNode;
  Connectivix::ConnectivityMatrix<Node, Node> *m_nodesOfCellsOfNode;

  Connectivix::ConnectivityMatrix<Edge, Face> *m_facesOfEdge;

  Connectivix::ConnectivityMatrix<Face, Face> *m_oppositeFaceOfFace;

private:
  void _computePressureAndCellPseudoViscosityForces();

private:
  void _specialInit();
  void _computeBoundaryMatrices();
  void _doCall(const char *func_name, std::function<void()> func);
  void computeGeometricValues2();

  void cellScalarPseudoViscosity();
  ARCCORE_HOST_DEVICE static inline void computeCQs(Real3 node_coord[8], Real3 face_coord[6], Span<Real3> cqs);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MicroHydroModule::MicroHydroModule(const ModuleBuildInfo &sbi)
    : ArcaneMicroHydroObject(sbi), m_time_stats(sbi.subDomain()->timeStats()), m_elapsed_timer(sbi.subDomain(), "MicroHydro", Timer::TimerReal),
      m_node_index_in_cells(MemoryUtils::getDefaultDataAllocator()) {}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::_doCall(const char *func_name, std::function<void()> func) {
  {
    Timer::Sentry ts_elapsed(&m_elapsed_timer);
    Timer::Action ts_action1(m_time_stats, func_name);
    func();
  }
}

#define DO_CALL(func_name) _doCall(#func_name, [&] { this->func_name(); })

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::hydroBuild() {
  info() << "Bench MicroHydro";
  Runner *r = acceleratorMng()->defaultRunner();
  if (r) {
    m_runner = *r;
  }
  m_default_queue = makeQueue(m_runner);
  Connectivity c(mesh()->connectivity());
  c.enableConnectivity(Connectivity::CT_HasEdge | Connectivity::CT_FaceToEdge | Connectivity::CT_EdgeToFace);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialisation du module hydro lors du démarrage du cas.
 */
void MicroHydroModule::hydroStartInit() {
  m_connectivity_view.setMesh(this->mesh());

  // Dimensionne les variables tableaux
  m_cell_cqs.resize(8);
  DO_CALL(_computeNodeIndexInCells);
  DO_CALL(_computeBoundaryMatrices);

  // Vérifie que les valeurs initiales sont correctes
  {
    Integer nb_error = 0;
    auto command = makeCommand(m_default_queue);
    auto in_pressure = ax::viewIn(command, m_pressure);
    auto in_adiabatic_cst = ax::viewIn(command, m_adiabatic_cst);
    ax::VariableCellRealInView in_density = ax::viewIn(command, m_density);
    ENUMERATE_CELL(icell, allCells()) {
      CellLocalId cid = *icell;
      Real pressure = in_pressure[cid];
      Real adiabatic_cst = in_adiabatic_cst[cid];
      Real density = in_density[cid];
      if (math::isZero(pressure) || math::isZero(density) || math::isZero(adiabatic_cst)) {
        info() << "Null valeur for cell=" << ItemPrinter(*icell) << " density=" << density << " pressure=" << pressure << " adiabatic_cst=" << adiabatic_cst;
        ++nb_error;
      }
    }
    if (nb_error != 0)
      ARCANE_FATAL("Some ({0}) cells are not initialised", nb_error);
  }

  // Initialise le delta-t
  Real deltat_init = options()->getDeltatInit();
  m_delta_t_n = deltat_init;
  m_delta_t_f = deltat_init;

  // Initialise les données géométriques: volume, cqs, longueurs
  // caractéristiques
  computeGeometricValues();

  m_node_mass.fill(ARCANE_REAL(0.0));
  m_velocity.fill(Real3::zero());

  // Initialisation de la masses des mailles et des masses nodale
  ENUMERATE_CELL(icell, allCells()) {
    Cell cell = *icell;
    m_cell_mass[icell] = m_density[icell] * m_volume[icell];

    Real contrib_node_mass = ARCANE_REAL(0.125) * m_cell_mass[cell];
    for (NodeEnumerator i_node(cell.nodes()); i_node.hasNext(); ++i_node) {
      m_node_mass[i_node] += contrib_node_mass;
    }
  }

  m_node_mass.synchronize();

  {
    info() << "Initialize SoundSpeed and InternalEnergy";
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    // Initialise l'énergie et la vitesse du son
    auto in_pressure = ax::viewIn(command, m_pressure);
    auto in_density = ax::viewIn(command, m_density);
    auto in_adiabatic_cst = ax::viewIn(command, m_adiabatic_cst);

    auto out_internal_energy = ax::viewOut(command, m_internal_energy);
    auto out_sound_speed = ax::viewOut(command, m_sound_speed);

    command << RUNCOMMAND_ENUMERATE(Cell, vi, allCells()) {
      Real pressure = in_pressure[vi];
      Real adiabatic_cst = in_adiabatic_cst[vi];
      Real density = in_density[vi];
      out_internal_energy[vi] = pressure / ((adiabatic_cst - 1.0) * density);
      out_sound_speed[vi] = math::sqrt(adiabatic_cst * pressure / density);
    };
  }

  // Remplit la structure contenant les informations sur les conditions aux
  // limites Cela permet de garantir avec les accélérateurs qu'on pourra accéder
  // de manière concurrente aux données.
  {
    m_boundary_conditions.clear();
    for (auto bc : options()->getBoundaryCondition()) {
      FaceGroup face_group = bc->getSurface();
      Real value = bc->getValue();
      MicroHydroTypes::eBoundaryCondition type = bc->getType();
      BoundaryCondition bcn;
      bcn.nodes = face_group.nodeGroup();
      bcn.value = value;
      bcn.type = type;
      m_boundary_conditions.add(bcn);
    }
  }

  options()->stOutput()->init();

  info() << "END_START_INIT";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Point d'entrée appelé après un équilibrage de charge.
 *
 * Il faut reconstruire les informations de connectivités propres à notre
 * module.
 */
void MicroHydroModule::hydroOnMeshChanged() {
  info() << "Hydro: OnMeshChanged";

  m_connectivity_view.setMesh(this->mesh());
  DO_CALL(_computeNodeIndexInCells);
  DO_CALL(_computeBoundaryMatrices);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des forces au temps courant \f$t^{n}\f$
 */
void MicroHydroModule::computeForces() {
  // Calcul pour chaque noeud de chaque maille la contribution
  // des forces de pression et de la pseudo-viscosite si necessaire

  Real linear_coef = options()->getViscosityLinearCoef();
  Real quadratic_coef = options()->getViscosityQuadraticCoef();

  // auto cnc = m_connectivity_view.cellNode();

  // Calcul de la divergence de la vitesse et de la viscosité scalaire
  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto in_density = viewIn(command, m_density);
    auto in_velocity = viewIn(command, m_velocity);
    auto in_caracteristic_length = viewIn(command, m_caracteristic_length);
    auto in_volume = viewIn(command, m_volume);
    auto in_sound_speed = viewIn(command, m_sound_speed);
    auto in_cell_cqs = viewIn(command, m_cell_cqs);
    auto out_cell_viscosity_force = viewOut(command, m_cell_viscosity_force);
    // command << RUNCOMMAND_ENUMERATE(Cell, cid, allCells()) {
    //   Real delta_speed = 0.0;

    //   Vector *nodesOfCell = Library::createVector(nbNode());
    //   nodesOfCell->extractRow(*m_nodesOfCell, cid);

    //   // Vector *facesOfCell = Library::createVector(nbFace());
    //   // facesOfCell->extractRow(*m_facesOfCell, cid);

    //   // for (cubool::index node : *nodesOfCell) {
    //   //   Vector *facesOfNode = Library::createVector(nbFace());
    //   //   facesOfNode->extractRow(*m_facesOfNode, node);
    //   //   Vector *facesOfNodeInCell = Library::createVector(nbFace());
    //   //   facesOfNodeInCell->eWiseMult(*facesOfNode, *facesOfCell, true);
    //   // }

    //   Int32 i = 0;
    //   for (cubool::index node : *nodesOfCell) {
    //     // Map to index of node in cell.
    //     delta_speed += math::dot(in_velocity[NodeLocalId(node)], in_cell_cqs[cid][i]);
    //     ++i;
    //   }
    //   delta_speed /= in_volume[cid];

    //   // Capture uniquement les chocs
    //   bool shock = (math::min(ARCANE_REAL(0.0), delta_speed) < ARCANE_REAL(0.0));
    //   if (shock) {
    //     Real rho = in_density[cid];
    //     Real sound_speed = in_sound_speed[cid];
    //     Real dx = in_caracteristic_length[cid];
    //     Real quadratic_viscosity = rho * dx * dx * delta_speed * delta_speed;
    //     Real linear_viscosity = -rho * sound_speed * dx * delta_speed;
    //     Real scalar_viscosity = linear_coef * linear_viscosity + quadratic_coef * quadratic_viscosity;
    //     out_cell_viscosity_force[cid] = scalar_viscosity;
    //   } else {
    //     out_cell_viscosity_force[cid] = 0.0;
    //   }
    // };
  }

  constexpr int max_node_cell = MAX_NODE_CELL;
  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto in_pressure = viewIn(command, m_pressure);
    auto in_cell_viscosity_force = viewIn(command, m_cell_viscosity_force);
    auto in_cell_cqs = viewIn(command, m_cell_cqs);
    auto out_force = viewOut(command, m_force);

    // command << RUNCOMMAND_ENUMERATE(Node, node, allNodes()) {
    //   Int32 first_pos = node.localId() * max_node_cell;
    //   Real3 force;
    //   Integer index = 0;

    //   Vector *cellsOfNode = Library::createVector(nbCell());
    //   cellsOfNode->extractRow(*m_cellsOfNode, node);

    //   for (cubool::index cell : *cellsOfNode) {
    //     // node index in cell = index of the node in its column vector in
    //     // nodesOfCell
    //     Int16 node_index = m_nodesOfCell->getIndexInRow(cell, node);
    //     CellLocalId cid = CellLocalId(cell);
    //     Real scalar_viscosity = in_cell_viscosity_force[cid];
    //     Real pressure = in_pressure[cid];
    //     force += (pressure + scalar_viscosity) * in_cell_cqs[cid][node_index];
    //     ++index;
    //   }

    //   out_force[node] = force;
    // };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul de l'impulsion (phase2).
 */
void MicroHydroModule::computeVelocity() {
  m_force.synchronize();

  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);
  auto in_node_mass = viewIn(command, m_node_mass);
  auto in_force = viewIn(command, m_force);
  auto in_out_velocity = viewInOut(command, m_velocity);
  Real delta_t_n = m_delta_t_n();

  // Calcule l'impulsion aux noeuds
  command << RUNCOMMAND_ENUMERATE(Node, node, allNodes()) {
    Real node_mass = in_node_mass[node];
    Real3 old_velocity = in_out_velocity[node];
    Real3 new_velocity = old_velocity + (delta_t_n / node_mass) * in_force[node];
    in_out_velocity[node] = new_velocity;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul de l'impulsion (phase3).
 */
void MicroHydroModule::computeViscosityWork() {
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);
  auto in_cell_viscosity_force = viewIn(command, m_cell_viscosity_force);
  auto in_velocity = viewIn(command, m_velocity);
  auto out_viscosity_work = viewOut(command, m_viscosity_work);
  auto in_cell_cqs = viewIn(command, m_cell_cqs);

  // // Calcul du travail des forces de viscosité dans une maille
  // command << RUNCOMMAND_ENUMERATE(Cell, cid, allCells()) {
  //   Vector *nodesOfCell = Library::createVector(nbNode());
  //   nodesOfCell->extractRow(*m_nodesOfCell, cid);

  //   Real work = 0.0;
  //   Real scalar_viscosity = in_cell_viscosity_force[cid];
  //   if (!math::isZero(scalar_viscosity)) {
  //     Integer i = 0;
  //     for (cubool::index node : *nodesOfCell) {
  //       work += math::dot(scalar_viscosity * in_cell_cqs[cid][i], in_velocity[NodeLocalId(node)]);
  //       ++i;
  //     }
  //   }
  //   out_viscosity_work[cid] = work;
  // };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Prise en compte des conditions aux limites.
 */
void MicroHydroModule::applyBoundaryCondition() {
  auto queue = makeQueue(m_runner);

  // Pour cette méthode, comme les conditions aux limites sont sur des groupes
  // indépendants (ou alors avec la même valeur si c'est sur les mêmes noeuds),
  // on peut exécuter les noyaux en asynchrone.
  queue.setAsync(true);

  // Repositionne les vues si les groupes associés ont été modifiés
  for (auto &bc : m_boundary_conditions)
    bc.view = bc.nodes.view();
  for (auto bc : m_boundary_conditions) {
    Real value = bc.value;
    MicroHydroTypes::eBoundaryCondition type = bc.type;
    NodeVectorView view = bc.view;

    auto command = makeCommand(queue);
    auto in_out_velocity = viewInOut(command, m_velocity);
    // boucle sur les faces de la surface
    command << RUNCOMMAND_ENUMERATE(Node, node, view) {
      // boucle sur les noeuds de la face
      Real3 v = in_out_velocity[node];
      switch (type) {
      case MicroHydroTypes::VelocityX:
        v.x = value;
        break;
      case MicroHydroTypes::VelocityY:
        v.y = value;
        break;
      case MicroHydroTypes::VelocityZ:
        v.z = value;
        break;
      case MicroHydroTypes::Unknown:
        break;
      }
      in_out_velocity[node] = v;
    };
  }
  queue.barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Déplace les noeuds.
 */
void MicroHydroModule::moveNodes() {
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);
  Real deltat_f = m_delta_t_f();

  auto in_velocity = viewIn(command, m_velocity);
  auto in_out_node_coord = viewInOut(command, m_node_coord);

  command << RUNCOMMAND_ENUMERATE(Node, node, allNodes()) {
    Real3 coord = in_out_node_coord[node];
    in_out_node_coord[node] = coord + (deltat_f * in_velocity[node]);
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mise à jour des densités et calcul de l'accroissements max
 *	  de la densité sur l'ensemble du maillage.
 */
void MicroHydroModule::updateDensity() {
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);
  ax::ReducerMax<double> density_ratio_maximum(command);
  density_ratio_maximum.setValue(0.0);
  auto in_cell_mass = viewIn(command, m_cell_mass);
  auto in_volume = viewIn(command, m_volume);
  auto in_out_density = viewInOut(command, m_density);

  command << RUNCOMMAND_ENUMERATE(Cell, cid, allCells()) {
    Real old_density = in_out_density[cid];
    Real new_density = in_cell_mass[cid] / in_volume[cid];

    in_out_density[cid] = new_density;

    Real density_ratio = (new_density - old_density) / new_density;

    density_ratio_maximum.max(density_ratio);
  };

  m_density_ratio_maximum = density_ratio_maximum.reduce();

  options()->stOutput()->addElementInRow("m_density_ratio_maximum", parallelMng()->reduce(Parallel::ReduceMax, m_density_ratio_maximum()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'équation d'état et calcul l'énergie interne et la
 * pression.
 */
void MicroHydroModule::applyEquationOfState() {
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);
  const Real deltatf = m_delta_t_f();
  const bool add_viscosity_force = true;

  auto in_adiabatic_cst = viewIn(command, m_adiabatic_cst);
  auto in_volume = viewIn(command, m_volume);
  auto in_density = viewIn(command, m_density);
  auto in_old_volume = viewIn(command, m_old_volume);
  auto in_cell_mass = viewIn(command, m_cell_mass);
  auto in_viscosity_work = viewIn(command, m_viscosity_work);

  auto in_out_internal_energy = viewInOut(command, m_internal_energy);
  auto out_sound_speed = viewOut(command, m_sound_speed);
  auto out_pressure = viewOut(command, m_pressure);

  // Calcul de l'énergie interne
  command << RUNCOMMAND_ENUMERATE(Cell, vi, allCells()) {
    Real adiabatic_cst = in_adiabatic_cst[vi];
    Real volume_ratio = in_volume[vi] / in_old_volume[vi];
    Real x = 0.5 * (adiabatic_cst - 1.0);
    Real numer_accrois_nrj = 1.0 + x * (1.0 - volume_ratio);
    Real denom_accrois_nrj = 1.0 + x * (1.0 - (1.0 / volume_ratio));
    Real internal_energy = in_out_internal_energy[vi];
    internal_energy = internal_energy * (numer_accrois_nrj / denom_accrois_nrj);

    // Prise en compte du travail des forces de viscosité
    if (add_viscosity_force)
      internal_energy = internal_energy - deltatf * in_viscosity_work[vi] / (in_cell_mass[vi] * denom_accrois_nrj);

    in_out_internal_energy[vi] = internal_energy;

    Real density = in_density[vi];
    Real pressure = (adiabatic_cst - 1.0) * density * internal_energy;
    Real sound_speed = math::sqrt(adiabatic_cst * pressure / density);
    out_pressure[vi] = pressure;
    out_sound_speed[vi] = sound_speed;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des nouveaux pas de temps.
 */
void MicroHydroModule::computeDeltaT() {
  const Real old_dt = m_global_deltat();

  // Calcul du pas de temps pour le respect du critère de CFL

  Real minimum_aux = FloatInfo<Real>::maxValue();

  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    ax::ReducerMin<double> minimum_aux_reducer(command);
    auto in_sound_speed = viewIn(command, m_sound_speed);
    auto in_caracteristic_length = viewIn(command, m_caracteristic_length);
    command << RUNCOMMAND_ENUMERATE(Cell, cid, allCells()) {
      Real cell_dx = in_caracteristic_length[cid];
      Real sound_speed = in_sound_speed[cid];
      Real dx_sound = cell_dx / sound_speed;
      minimum_aux_reducer.min(dx_sound);
    };
    minimum_aux = minimum_aux_reducer.reduce();
  }

  Real new_dt = options()->getCfl() * minimum_aux;

  // Pas de variations trop brutales à la hausse comme à la baisse
  Real max_dt = (ARCANE_REAL(1.0) + options()->getVariationSup()) * old_dt;
  Real min_dt = (ARCANE_REAL(1.0) - options()->getVariationInf()) * old_dt;

  new_dt = math::min(new_dt, max_dt);
  new_dt = math::max(new_dt, min_dt);

  Real max_density_ratio = m_density_ratio_maximum();

  // control de l'accroissement relatif de la densité
  Real dgr = options()->getDensityGlobalRatio();
  if (max_density_ratio > dgr)
    new_dt = math::min(old_dt * dgr / max_density_ratio, new_dt);

  IParallelMng *pm = mesh()->parallelMng();
  new_dt = pm->reduce(Parallel::ReduceMin, new_dt);

  // Respect des valeurs min et max imposées par le fichier de données .plt
  new_dt = math::min(new_dt, options()->getDeltatMax());
  new_dt = math::max(new_dt, options()->getDeltatMin());

  options()->stOutput()->addElementInRow("new_dt", new_dt);

  // Le dernier calcul se fait exactement au temps stopTime()
  {
    Real stop_time = options()->getFinalTime();
    bool not_yet_finish = (m_global_time() < stop_time);
    bool too_much = ((m_global_time() + new_dt) > stop_time);

    if (not_yet_finish && too_much) {
      new_dt = stop_time - m_global_time();
      subDomain()->timeLoopMng()->stopComputeLoop(true);
    }
  }

  // Mise à jour des variables
  m_old_dt_f.assign(old_dt);
  m_delta_t_n.assign(ARCANE_REAL(0.5) * (old_dt + new_dt));
  m_delta_t_f.assign(new_dt);
  m_global_deltat.assign(new_dt);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des résultantes aux noeuds d'une maille hexaédrique.
 *
 * La méthode utilisée est celle du découpage en quatre triangles.
 */
inline void MicroHydroModule::computeCQs(Real3 node_coord[8], Real3 face_coord[6], Span<Real3> cqs) {
  const Real3 c0 = face_coord[0];
  const Real3 c1 = face_coord[1];
  const Real3 c2 = face_coord[2];
  const Real3 c3 = face_coord[3];
  const Real3 c4 = face_coord[4];
  const Real3 c5 = face_coord[5];

  const Real demi = ARCANE_REAL(0.5);
  const Real five = ARCANE_REAL(5.0);

  // Calcul des normales face 1 :
  const Real3 n1a04 = demi * math::cross(node_coord[0] - c0, node_coord[3] - c0);
  const Real3 n1a03 = demi * math::cross(node_coord[3] - c0, node_coord[2] - c0);
  const Real3 n1a02 = demi * math::cross(node_coord[2] - c0, node_coord[1] - c0);
  const Real3 n1a01 = demi * math::cross(node_coord[1] - c0, node_coord[0] - c0);

  // Calcul des normales face 2 :
  const Real3 n2a05 = demi * math::cross(node_coord[0] - c1, node_coord[4] - c1);
  const Real3 n2a12 = demi * math::cross(node_coord[4] - c1, node_coord[7] - c1);
  const Real3 n2a08 = demi * math::cross(node_coord[7] - c1, node_coord[3] - c1);
  const Real3 n2a04 = demi * math::cross(node_coord[3] - c1, node_coord[0] - c1);

  // Calcul des normales face 3 :
  const Real3 n3a01 = demi * math::cross(node_coord[0] - c2, node_coord[1] - c2);
  const Real3 n3a06 = demi * math::cross(node_coord[1] - c2, node_coord[5] - c2);
  const Real3 n3a09 = demi * math::cross(node_coord[5] - c2, node_coord[4] - c2);
  const Real3 n3a05 = demi * math::cross(node_coord[4] - c2, node_coord[0] - c2);

  // Calcul des normales face 4 :
  const Real3 n4a09 = demi * math::cross(node_coord[4] - c3, node_coord[5] - c3);
  const Real3 n4a10 = demi * math::cross(node_coord[5] - c3, node_coord[6] - c3);
  const Real3 n4a11 = demi * math::cross(node_coord[6] - c3, node_coord[7] - c3);
  const Real3 n4a12 = demi * math::cross(node_coord[7] - c3, node_coord[4] - c3);

  // Calcul des normales face 5 :
  const Real3 n5a02 = demi * math::cross(node_coord[1] - c4, node_coord[2] - c4);
  const Real3 n5a07 = demi * math::cross(node_coord[2] - c4, node_coord[6] - c4);
  const Real3 n5a10 = demi * math::cross(node_coord[6] - c4, node_coord[5] - c4);
  const Real3 n5a06 = demi * math::cross(node_coord[5] - c4, node_coord[1] - c4);

  // Calcul des normales face 6 :
  const Real3 n6a03 = demi * math::cross(node_coord[2] - c5, node_coord[3] - c5);
  const Real3 n6a08 = demi * math::cross(node_coord[3] - c5, node_coord[7] - c5);
  const Real3 n6a11 = demi * math::cross(node_coord[7] - c5, node_coord[6] - c5);
  const Real3 n6a07 = demi * math::cross(node_coord[6] - c5, node_coord[2] - c5);

  const Real real_1div12 = ARCANE_REAL(1.0) / ARCANE_REAL(12.0);

  // Calcul des résultantes aux sommets :
  cqs[0] = (five * (n1a01 + n1a04 + n2a04 + n2a05 + n3a05 + n3a01) + (n1a02 + n1a03 + n2a08 + n2a12 + n3a06 + n3a09)) * real_1div12;
  cqs[1] = (five * (n1a01 + n1a02 + n3a01 + n3a06 + n5a06 + n5a02) + (n1a04 + n1a03 + n3a09 + n3a05 + n5a10 + n5a07)) * real_1div12;
  cqs[2] = (five * (n1a02 + n1a03 + n5a07 + n5a02 + n6a07 + n6a03) + (n1a01 + n1a04 + n5a06 + n5a10 + n6a11 + n6a08)) * real_1div12;
  cqs[3] = (five * (n1a03 + n1a04 + n2a08 + n2a04 + n6a08 + n6a03) + (n1a01 + n1a02 + n2a05 + n2a12 + n6a07 + n6a11)) * real_1div12;
  cqs[4] = (five * (n2a05 + n2a12 + n3a05 + n3a09 + n4a09 + n4a12) + (n2a08 + n2a04 + n3a01 + n3a06 + n4a10 + n4a11)) * real_1div12;
  cqs[5] = (five * (n3a06 + n3a09 + n4a09 + n4a10 + n5a10 + n5a06) + (n3a01 + n3a05 + n4a12 + n4a11 + n5a07 + n5a02)) * real_1div12;
  cqs[6] = (five * (n4a11 + n4a10 + n5a10 + n5a07 + n6a07 + n6a11) + (n4a12 + n4a09 + n5a06 + n5a02 + n6a03 + n6a08)) * real_1div12;
  cqs[7] = (five * (n2a08 + n2a12 + n4a12 + n4a11 + n6a11 + n6a08) + (n2a04 + n2a05 + n4a09 + n4a10 + n6a07 + n6a03)) * real_1div12;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul du volume des mailles, des longueurs caractéristiques
 * et des résultantes aux sommets.
 */
void MicroHydroModule::computeGeometricValues() {
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);
  auto in_node_coord = viewIn(command, m_node_coord);
  auto in_out_cell_cqs = viewInOut(command, m_cell_cqs);
  auto in_volume = viewIn(command, m_volume);

  auto out_volume = viewOut(command, m_volume);
  auto out_old_volume = viewOut(command, m_old_volume);
  auto out_caracteristic_length = viewOut(command, m_caracteristic_length);

  // command << RUNCOMMAND_ENUMERATE(Cell, cid, allCells()) {
  //   Vector *nodesOfCell = Library::createVector(nbNode());
  //   nodesOfCell->extractRow(*m_nodesOfCell, cid);

  //   Vector *facesOfCell = Library::createVector(nbFace());
  //   facesOfCell->extractRow(*m_facesOfCell, cid);

  //   Matrix *edgesOfFaceInCell = Library::createMatrix(nbFace(), nbEdge());
  //   edgesOfFaceInCell->multiply(*m_edgesOfFace, *facesOfCell, false, true);
  //   Real3 normals[edgesOfFaceInCell->getNbVals()];

  //   Integer nb_faces = facesOfCell->getNbVals();
  //   Real3 face_coord[nb_faces];
  //   Real3 cell_center(0.0, 0.0, 0.0);
  //   cubool::index face_ids[nb_faces];

  //   {
  //     // Coordonnées des centres des faces
  //     Integer i = 0;
  //     for (cubool::index face : *facesOfCell) {
  //       Vector *nodesOfFace = Library::createVector(nbNode());
  //       nodesOfFace->extractRow(*m_nodesOfFace, face);
  //       Real3 tmp_face_center;
  //       for (cubool::index node : *nodesOfFace) {
  //         Real3 node_coord = in_node_coord[NodeLocalId(node)];
  //         tmp_face_center += node_coord;
  //       }
  //       face_coord[i] = 0.25 * tmp_face_center;
  //       face_ids[i] = face;
  //       i++;
  //     }
  //   }

  //   {
  //     // Coordonnées du centre de la maille
  //     Real3 tmp_cell_center;
  //     for (cubool::index node : *nodesOfCell) {
  //       tmp_cell_center += in_node_coord[NodeLocalId(node)];
  //     }
  //     cell_center = 0.125 * tmp_cell_center;
  //   }

  //   // Calcule la longueur caractéristique de la maille.
  //   {
  //     cubool::index firstNodeOfCell = *(nodesOfCell->begin());

  //     Vector *facesOfNode = Library::createVector(nbFace());
  //     facesOfNode->extractRow(*m_facesOfNode, firstNodeOfCell);
  //     Vector *facesOfNodeInCell = Library::createVector(nbFace());
  //     facesOfNodeInCell->eWiseMult(*facesOfNode, *facesOfCell, true);

  //     Real distances[facesOfNodeInCell->getNbVals()];
  //     Integer nb_medians = 0;
  //     cubool::index *face_iterator;
  //     Vector *oppositeFacesOfFace = Library::createVector(nbFace());
  //     Vector *oppositeFacesOfFaceInCell = Library::createVector(nbFace());
  //     for (cubool::index face : *facesOfNodeInCell) {
  //       oppositeFacesOfFace->extractRow(*m_oppositeFaceOfFace, face);
  //       oppositeFacesOfFaceInCell->eWiseMult(*oppositeFacesOfFace, *facesOfCell, true);
  //       cubool::index opposite_face = *(oppositeFacesOfFaceInCell->begin());

  //       Integer face_idx = m_facesOfCell->getIndexInRow(cid, face);
  //       Integer opposite_face_idx = m_facesOfCell->getIndexInRow(cid, opposite_face);

  //       Real3 median = face_coord[face_idx] - face_coord[opposite_face_idx];
  //       distances[nb_medians] = median.normL2();
  //       nb_medians++;
  //     }

  //     Real dx_numerator = 1.0;
  //     Real dx_denominator = 0.0;
  //     for (int i = 0; i < nb_medians - 1; ++i) {
  //       dx_numerator *= distances[i];
  //       for (int j = i + 1; j < nb_medians; ++j) {
  //         dx_denominator += distances[i] * distances[j];
  //       }
  //     }
  //     dx_numerator *= distances[nb_medians - 1];

  //     out_caracteristic_length[cid] = dx_numerator / dx_denominator;
  //   }

  //   // Calcule les résultantes aux sommets
  //   {
  //     // computeCQs(coord, face_coord, in_out_cell_cqs[cid]);

  //     // This matrix is hypersparse
  //     Matrix *facesOfEdgesInCell = Library::createMatrix(nbEdge(), nbFace());

  //     Vector *edgesOfNode = Library::createVector(nbEdge());
  //     Vector *edgesOfFace = Library::createVector(nbEdge());
  //     Vector *facesOfNode = Library::createVector(nbFace());

  //     Vector *adjacentEdges = Library::createVector(nbFace());
  //     Vector *oppositeEdges = Library::createVector(nbFace());

  //     for (cubool::index node : *nodesOfCell) {
  //       facesOfNode->extractRow(*m_facesOfNode, node);
  //       edgesOfNode->extractRow(*m_edgesOfNode, node);
  //       for (cubool::index face : *facesOfNode) {
  //         Integer face_idx = m_facesOfCell->getIndexInRow(cid, face);
  //         Real3 centerOfFace = face_coord[face_idx];

  //         edgesOfFace->extractRow(*m_edgesOfFace, face);

  //         // Intersecting the edges of the face and of the node
  //         adjacentEdges->eWiseMult(*edgesOfFace, *edgesOfNode, true);
  //         // Subtracting the edges of the node from the edges of the face
  //         oppositeEdges->eWiseSub(*edgesOfFace, *edgesOfNode, true);

  //         for (cubool::index edge : *adjacentEdges) {
  //         }
  //       }
  //     }
  //   }

  //   Span<const Real3> in_cqs(in_out_cell_cqs[cid]);

  //   // Calcule le volume de la maille
  //   {
  //     Real volume = 0.0;
  //     for (cubool::index node : *nodesOfCell) {
  //       volume += math::dot(in_node_coord[NodeLocalId(node)], in_cqs[NodeLocalId(node)]);
  //     }
  //     volume /= 3.0;

  //     out_old_volume[cid] = in_volume[cid];
  //     out_volume[cid] = volume;
  //   }
  // };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::hydroInit() {
  info() << "INIT: DTmin=" << options()->getDeltatMin() << " DTmax=" << options()->getDeltatMax() << " DT=" << m_global_deltat();
  if (m_global_deltat() > options()->getDeltatMax())
    ARCANE_FATAL("DeltaT > DTMax");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Connectivix::CSR *assembleCSR(const Int32 M, const Int32 N, const Int32 *rows, const Int32 *cols, const Int32 nnz) {
  Connectivix::CSR *result = new Connectivix::CSR(M, N);
  result->fromCoordinates(rows, cols, nnz);
  return result;
}

void MicroHydroModule::_computeBoundaryMatrices() {
  if (nbEdge() > 0) {
    info() << "Computing edges of node: " << nbNode() << " x " << nbEdge();

    m_edgesOfNode = new Connectivix::ConnectivityMatrix<Node, Edge>(nbNode(), nbEdge());
    m_edgesOfNode->build(m_connectivity_view.nodeEdge(), allNodes());

    m_nodesOfEdge = new Connectivix::ConnectivityMatrix<Edge, Node>(nbEdge(), nbNode());
    m_nodesOfEdge->build(m_connectivity_view.edgeNode(), allEdges());

    info() << "Number of elements: " << m_edgesOfNode->getNbVals();
    info() << "Sparsity: " << m_edgesOfNode->getNbVals() / (1.0 * m_edgesOfNode->getNbRows() * m_edgesOfNode->getNbCols());

    info() << "Computing faces of edge: " << nbEdge() << " x " << nbFace();

    m_facesOfEdge = new Connectivix::ConnectivityMatrix<Edge, Face>(nbEdge(), nbFace());
    m_facesOfEdge->build(m_connectivity_view.edgeFace(), allEdges());

    m_edgesOfFace = new Connectivix::ConnectivityMatrix<Face, Edge>(nbFace(), nbEdge());
    m_edgesOfFace->build(m_connectivity_view.faceEdge(), allFaces());

    info() << "Number of elements: " << m_facesOfEdge->getNbVals();
    info() << "Sparsity: " << m_facesOfEdge->getNbVals() / (1.0 * m_facesOfEdge->getNbRows() * m_facesOfEdge->getNbCols());
  } else {
    info() << "Computing faces of node: " << nbNode() << " x " << nbFace();

    m_facesOfNode = new Connectivix::ConnectivityMatrix<Node, Face>(nbNode(), nbFace());
    m_facesOfNode->build(m_connectivity_view.nodeFace(), allNodes());

    m_nodesOfFace = new Connectivix::ConnectivityMatrix<Face, Node>(nbFace(), nbNode());
    m_nodesOfFace->build(m_connectivity_view.faceNode(), allFaces());

    info() << "Number of elements: " << m_facesOfNode->getNbVals();
    info() << "Sparsity: " << m_facesOfNode->getNbVals() / (1.0 * m_facesOfNode->getNbRows() * m_facesOfNode->getNbCols());
  }

  info() << "Computing cells of face: " << nbFace() << " x " << nbCell();

  m_cellsOfFace = new Connectivix::ConnectivityMatrix<Face, Cell>(nbFace(), nbCell());
  m_cellsOfFace->build(m_connectivity_view.faceCell(), allFaces());

  m_facesOfCell = new Connectivix::ConnectivityMatrix<Cell, Face>(nbCell(), nbFace());
  m_facesOfCell->build(m_connectivity_view.cellFace(), allCells());

  info() << "Number of elements: " << m_cellsOfFace->getNbVals();
  info() << "Sparsity: " << m_cellsOfFace->getNbVals() / (1.0 * m_cellsOfFace->getNbRows() * m_cellsOfFace->getNbCols());

  if (nbEdge() > 0) {
    info() << "Computing faces of node: " << nbNode() << " x " << nbFace();

    m_facesOfNode = m_edgesOfNode->matMul(*m_facesOfEdge, m_runner);
    m_nodesOfFace = m_edgesOfFace->matMul(*m_nodesOfEdge, m_runner);

    info() << "Number of elements: " << m_facesOfNode->getNbVals();
    info() << "Sparsity: " << m_facesOfNode->getNbVals() / (1.0 * m_facesOfNode->getNbRows() * m_facesOfNode->getNbCols());
  }

  info() << "Computing cells of node: " << nbNode() << " x " << nbCell();

  m_cellsOfNode = m_facesOfNode->matMul(*m_cellsOfFace, m_runner);
  m_nodesOfCell = m_facesOfCell->matMul(*m_nodesOfFace, m_runner);

  info() << "Number of elements: " << m_cellsOfNode->getNbVals();
  info() << "Sparsity: " << m_cellsOfNode->getNbVals() / (1.0 * m_cellsOfNode->getNbRows() * m_cellsOfNode->getNbCols());

  {
    ofstream cellsOfNodeDump("./mtx/cells_of_node_indirect.mtx");
    m_cellsOfNode->m_data->dumpMatrix(cellsOfNodeDump);
    cellsOfNodeDump.close();
  }

  {
    Connectivix::ConnectivityMatrix<Node, Cell> checkCellsOfNode(nbNode(), nbCell());
    checkCellsOfNode.build(m_connectivity_view.nodeCell(), allNodes());
    ofstream cellsOfNodeDump("./mtx/cells_of_node_direct.mtx");
    checkCellsOfNode.m_data->dumpMatrix(cellsOfNodeDump);
    cellsOfNodeDump.close();
  }
  // Matrix *facesOfCellsOfNode = Library::createMatrix(nbNode(), nbFace());
  // facesOfCellsOfNode->multiply(*m_cellsOfNode, *m_facesOfCell, true, true);

  // m_nodesOfCellsOfNode = Library::createMatrix(nbNode(), nbNode());
  // m_nodesOfCellsOfNode->multiply(*facesOfCellsOfNode, *m_nodesOfFace, true, true);
  // info() << "Number of elements: " << m_nodesOfCellsOfNode->getNbVals();
  // info() << "Sparsity: " << m_nodesOfCellsOfNode->getNbVals() / (1.0 * m_nodesOfCellsOfNode->getNbRows() * m_nodesOfCellsOfNode->getNbCols());

  // info() << "Computing opposite faces: " << nbFace() << " x " << nbFace();
  // Matrix *faceNodeReachability1 = Library::createMatrix(nbFace(), nbFace());
  // faceNodeReachability1->multiply(*m_nodesOfFace, *m_facesOfNode, false,
  // true);

  // Matrix *faceNodeReachability1Transpose =
  //     Library::createMatrix(nbFace(), nbFace());
  // faceNodeReachability1Transpose->transpose(*faceNodeReachability1, true);

  // Matrix *faceNodeReachability2 = Library::createMatrix(nbFace(), nbFace());
  // faceNodeReachability2->multiply(*faceNodeReachability1,
  //                                 *faceNodeReachability1Transpose, false,
  //                                 true);

  // Matrix *faceNodeReachabilityExactly2 =
  //     Library::createMatrix(nbFace(), nbFace());
  // faceNodeReachabilityExactly2->eWiseSub(*faceNodeReachability2,
  //                                        *faceNodeReachability1, true);

  // Matrix *faceCellReachability1 = Library::createMatrix(nbFace(), nbFace());
  // faceCellReachability1->multiply(*m_cellsOfFace, *m_facesOfCell, false,
  // true);

  // m_oppositeFaceOfFace = Library::createMatrix(nbFace(), nbFace());
  // m_oppositeFaceOfFace->eWiseMult(*faceNodeReachabilityExactly2,
  //                                 *faceCellReachability1, true);

  // info() << "Number of elements: " << m_oppositeFaceOfFace->getNbVals();
  // info() << "Sparsity: "
  //        << m_oppositeFaceOfFace->getNbVals() /
  //               (1.0 * m_oppositeFaceOfFace->getNbRows() *
  //                m_oppositeFaceOfFace->getNbCols());

  {
    ofstream nodesOfCellDump("./mtx/nodes_of_cell.mtx");
    m_nodesOfCell->m_data->dumpMatrix(nodesOfCellDump);
    nodesOfCellDump.close();
  }
  {
    Connectivix::ConnectivityMatrix<Node, Cell> *transposed = m_nodesOfCell->transpose(m_runner);
    ofstream nodesOfCellTransposeDump("./mtx/nodes_of_cell_from_transpose.mtx");
    transposed->m_data->dumpMatrix(nodesOfCellTransposeDump);
    nodesOfCellTransposeDump.close();
  }

  ofstream facesOfCellDump("./mtx/faces_of_cell.mtx");
  m_facesOfCell->m_data->dumpMatrix(facesOfCellDump);
  facesOfCellDump.close();

  if (nbEdge() > 0) {
    ofstream edgesOfFaceDump("./mtx/edges_of_face.mtx");
    m_edgesOfFace->m_data->dumpMatrix(edgesOfFaceDump);
    edgesOfFaceDump.close();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::_computeNodeIndexInCells() {
  info() << "ComputeNodeIndexInCells with accelerator";
  // Un nœud est connecté au maximum à MAX_NODE_CELL mailles
  // Calcule pour chaque nœud son index dans chacune des
  // mailles à laquelle il est connecté.
  NodeGroup nodes = allNodes();
  Integer nb_node = nodes.size();
  m_node_index_in_cells.resize(MAX_NODE_CELL * nb_node);

  auto node_cell_cty = m_connectivity_view.nodeCell();
  auto cell_node_cty = m_connectivity_view.cellNode();

  auto command = makeCommand(m_default_queue);
  auto inout_node_index_in_cells = m_node_index_in_cells.span();

  command << RUNCOMMAND_ENUMERATE(Node, node, nodes) {
    Int32 first_pos = node.localId() * MAX_NODE_CELL;

    Int32 index = 0;
    for (CellLocalId cell : node_cell_cty.cells(node)) {
      Int8 node_index_in_cell = 0;
      for (NodeLocalId cell_node : cell_node_cty.nodes(cell)) {
        if (cell_node == node)
          break;
        ++node_index_in_cell;
      }
      inout_node_index_in_cells[first_pos + index] = node_index_in_cell;
      ++index;
    }

    // Remplit avec la valeur nulle les derniers éléments
    for (; index < MAX_NODE_CELL; ++index)
      inout_node_index_in_cells[first_pos + index] = -1;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::hydroExit() {

  delete m_facesOfNode;
  delete m_cellsOfFace;
  delete m_nodesOfFace;
  delete m_facesOfCell;
  delete m_nodesOfCell;
  delete m_cellsOfNode;
  delete m_nodesOfCellsOfNode;
  delete m_oppositeFaceOfFace;

  info() << "Hydro exit entry point";
  m_time_stats->dumpCurrentStats("SH_DoOneIteration");

  // On ajoute un argument de ligne de commande pour changer le répertoire
  // des fichiers de références.
  String reference_input = subDomain()->applicationInfo().commandLineArguments().getParameter("ReferenceDirectory");

  // Si l'on veut comparer les valeurs.
  if (options()->getCheckNumericalResult() || !reference_input.empty()) {

    // On ajoute un arguments de ligne de commande pour déterminer si lecture ou
    // écriture.
    bool overwrite_reference = (subDomain()->applicationInfo().commandLineArguments().getParameter("OverwriteReference") == "true");

    // On initialise le comparateur.
    options()->stComparator()->init(options()->stOutput());

    // Si l'utilisateur veut un autre emplacement pour les fichiers de
    // références.
    if (!reference_input.empty()) {
      info() << "Set reference directory: " << reference_input;
      options()->stComparator()->editRootDirectory(Directory(reference_input));
    }

    // Si demande d'écriture.
    if (overwrite_reference) {
      info() << "Write reference file";
      options()->stComparator()->writeReferenceFile(0);
    }

    // Sinon lecture.
    else {
      // Si le fichier existe, comparaison.
      if (options()->stComparator()->isReferenceExist(0)) {
        options()->stComparator()->addEpsilonRow("m_density_ratio_maximum", 1.0e-10);
        options()->stComparator()->addEpsilonRow("new_dt", 1.0e-13);
        if (options()->stComparator()->compareWithReference(0)) {
          info() << "Comparator: OK";
        } else {
          ARCANE_FATAL("Comparator: NOK");
        }
      } else {
        ARCANE_FATAL("Ref file not found");
      }
    }
  }

  // // Précision maximum.
  // options()->stOutput()->setPrecision(std::numeric_limits<Real>::max_digits10);
  // options()->stOutput()->setForcedToUseScientificNotation(true);
  // options()->stOutput()->setFixed(false);

  // On écrit les valeurs de l'exécution actuelle.
  options()->stOutput()->writeFile(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MicroHydroModule::doOneIteration() {
  options()->stOutput()->addColumn("Iteration " + String::fromNumber(m_global_iteration()));
  DO_CALL(computeForces);
  DO_CALL(computeVelocity);
  DO_CALL(computeViscosityWork);
  DO_CALL(applyBoundaryCondition);
  DO_CALL(moveNodes);
  DO_CALL(computeGeometricValues);
  DO_CALL(updateDensity);
  DO_CALL(applyEquationOfState);
  DO_CALL(computeDeltaT);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(MicroHydroModule, MicroHydro);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MicroHydro

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
