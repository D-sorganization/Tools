use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::f64::consts::PI;

const MM_PER_M: f64 = 1000.0;

#[pyclass(module = "tools_core.electrode_advisor", get_all)]
#[derive(Clone, Debug)]
pub struct BathDefaults {
    pub shape: String,
    pub width_m: f64,
    pub depth_m: f64,
    pub height_m: f64,
    pub glass_level_m: f64,
}

#[pymethods]
impl BathDefaults {
    #[new]
    #[pyo3(signature = (shape="rectangular".to_string(), width_m=3.0, depth_m=2.0, height_m=2.5, glass_level_m=1.5))]
    pub fn new(
        shape: String,
        width_m: f64,
        depth_m: f64,
        height_m: f64,
        glass_level_m: f64,
    ) -> Self {
        Self {
            shape,
            width_m,
            depth_m,
            height_m,
            glass_level_m,
        }
    }
}

#[pyclass(module = "tools_core.electrode_advisor", get_all)]
#[derive(Clone, Debug)]
pub struct ElectrodeDefaults {
    pub electrode_type: String,
    pub count: usize,
    pub top_offset_m: f64,
    pub current_length_mm: f64,
    pub worn_length_mm: f64,
    pub diameter_mm: f64,
    pub operating_current_a: f64,
    pub plasma_temperature_c: f64,
}

#[pymethods]
impl ElectrodeDefaults {
    #[new]
    #[pyo3(signature = (electrode_type="graphite_standard".to_string(), count=3, top_offset_m=0.1, current_length_mm=1500.0, worn_length_mm=150.0, diameter_mm=150.0, operating_current_a=2500.0, plasma_temperature_c=1500.0))]
    pub fn new(
        electrode_type: String,
        count: usize,
        top_offset_m: f64,
        current_length_mm: f64,
        worn_length_mm: f64,
        diameter_mm: f64,
        operating_current_a: f64,
        plasma_temperature_c: f64,
    ) -> Self {
        Self {
            electrode_type,
            count,
            top_offset_m,
            current_length_mm,
            worn_length_mm,
            diameter_mm,
            operating_current_a,
            plasma_temperature_c,
        }
    }
}

#[pyclass(module = "tools_core.electrode_advisor", get_all)]
#[derive(Clone, Debug)]
pub struct DraftingEnvelope {
    pub bath_shell_thickness_mm: f64,
    pub glass_clearance_mm: f64,
    pub electrode_holder_height_mm: f64,
    pub electrode_holder_radius_factor: f64,
    pub tip_band_height_mm: f64,
}

#[pymethods]
impl DraftingEnvelope {
    #[new]
    #[pyo3(signature = (bath_shell_thickness_mm=25.0, glass_clearance_mm=10.0, electrode_holder_height_mm=100.0, electrode_holder_radius_factor=2.0, tip_band_height_mm=20.0))]
    pub fn new(
        bath_shell_thickness_mm: f64,
        glass_clearance_mm: f64,
        electrode_holder_height_mm: f64,
        electrode_holder_radius_factor: f64,
        tip_band_height_mm: f64,
    ) -> Self {
        Self {
            bath_shell_thickness_mm,
            glass_clearance_mm,
            electrode_holder_height_mm,
            electrode_holder_radius_factor,
            tip_band_height_mm,
        }
    }
}

#[pyclass(module = "tools_core.electrode_advisor", get_all)]
#[derive(Clone, Debug)]
pub struct ElectrodePlacement {
    pub index: usize,
    pub angle_radians: f64,
    pub viewer_x_m: f64,
    pub viewer_y_m: f64,
    pub viewer_z_m: f64,
    pub cad_x_mm: f64,
    pub cad_y_mm: f64,
    pub cad_z_mm: f64,
    pub diameter_mm: f64,
    pub current_a: f64,
    pub current_length_mm: f64,
    pub worn_length_mm: f64,
    pub effective_length_mm: f64,
}

#[pyclass(module = "tools_core.electrode_advisor", get_all)]
#[derive(Clone, Debug)]
pub struct ElectrodeAdvisorLayout {
    pub bath: BathDefaults,
    pub electrodes: ElectrodeDefaults,
    pub drafting: DraftingEnvelope,
    pub placements: Vec<ElectrodePlacement>,
}

#[pymethods]
impl ElectrodeAdvisorLayout {
    #[getter]
    pub fn bath_width_mm(&self) -> f64 {
        self.bath.width_m * MM_PER_M
    }

    #[getter]
    pub fn bath_depth_mm(&self) -> f64 {
        self.bath.depth_m * MM_PER_M
    }

    #[getter]
    pub fn bath_height_mm(&self) -> f64 {
        self.bath.height_m * MM_PER_M
    }

    #[getter]
    pub fn glass_level_mm(&self) -> f64 {
        self.bath.glass_level_m * MM_PER_M
    }

    pub fn to_manifest(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyDict, PyList};

        let bath = PyDict::new(py);
        bath.set_item("shape", &self.bath.shape)?;
        bath.set_item("width_m", self.bath.width_m)?;
        bath.set_item("depth_m", self.bath.depth_m)?;
        bath.set_item("height_m", self.bath.height_m)?;
        bath.set_item("glass_level_m", self.bath.glass_level_m)?;

        let electrodes = PyDict::new(py);
        electrodes.set_item("type", &self.electrodes.electrode_type)?;
        electrodes.set_item("count", self.electrodes.count)?;
        electrodes.set_item("top_offset_m", self.electrodes.top_offset_m)?;
        electrodes.set_item("current_length_mm", self.electrodes.current_length_mm)?;
        electrodes.set_item("worn_length_mm", self.electrodes.worn_length_mm)?;
        electrodes.set_item("diameter_mm", self.electrodes.diameter_mm)?;
        electrodes.set_item("operating_current_a", self.electrodes.operating_current_a)?;
        electrodes.set_item("plasma_temperature_c", self.electrodes.plasma_temperature_c)?;

        let drafting = PyDict::new(py);
        drafting.set_item(
            "bath_shell_thickness_mm",
            self.drafting.bath_shell_thickness_mm,
        )?;
        drafting.set_item("glass_clearance_mm", self.drafting.glass_clearance_mm)?;
        drafting.set_item(
            "electrode_holder_height_mm",
            self.drafting.electrode_holder_height_mm,
        )?;
        drafting.set_item(
            "electrode_holder_radius_factor",
            self.drafting.electrode_holder_radius_factor,
        )?;
        drafting.set_item("tip_band_height_mm", self.drafting.tip_band_height_mm)?;

        let placements = PyList::empty(py);
        for p in &self.placements {
            let p_dict = PyDict::new(py);
            p_dict.set_item("index", p.index)?;
            p_dict.set_item("angle_radians", p.angle_radians)?;

            let pos_m = PyList::new(py, &[p.viewer_x_m, p.viewer_y_m, p.viewer_z_m])?;
            p_dict.set_item("viewer_position_m", pos_m)?;

            let pos_mm = PyList::new(py, &[p.cad_x_mm, p.cad_y_mm, p.cad_z_mm])?;
            p_dict.set_item("cad_position_mm", pos_mm)?;

            p_dict.set_item("effective_length_mm", p.effective_length_mm)?;
            p_dict.set_item("current_a", p.current_a)?;

            placements.append(p_dict)?;
        }

        let source_viewer = PyDict::new(py);
        source_viewer.set_item(
            "component",
            "Tools/src/electrode_advisor/web/src/components/GlassBath3DViewer.tsx",
        )?;
        source_viewer.set_item(
            "calculator",
            "Tools/src/electrode_advisor/web/src/components/ElectrodeAdvisorCalculator.tsx",
        )?;

        let root = PyDict::new(py);
        root.set_item("project", "electrode_advisor_default_layout")?;
        root.set_item("source_viewer", source_viewer)?;
        root.set_item("bath", bath)?;
        root.set_item("electrodes", electrodes)?;
        root.set_item("drafting_assumptions", drafting)?;
        root.set_item("placements", placements)?;

        Ok(root.into_any().unbind())
    }
}

#[pyfunction]
pub fn build_default_placements(
    bath: &BathDefaults,
    electrodes: &ElectrodeDefaults,
) -> Vec<ElectrodePlacement> {
    let spacing_m = bath.width_m * 0.6;
    let radius_m = spacing_m * 0.4;
    let cad_z_mm = (bath.height_m + electrodes.top_offset_m) * MM_PER_M;
    let current_per_electrode = electrodes.operating_current_a / (electrodes.count as f64);
    let effective_length_mm = electrodes.current_length_mm - electrodes.worn_length_mm;

    let mut placements = Vec::with_capacity(electrodes.count);
    for index in 0..electrodes.count {
        let angle = (index as f64 / electrodes.count as f64) * 2.0 * PI;
        let viewer_x_m = if electrodes.count == 1 {
            0.0
        } else {
            angle.cos() * radius_m
        };
        let viewer_z_m = if electrodes.count == 1 {
            0.0
        } else {
            angle.sin() * radius_m
        };

        placements.push(ElectrodePlacement {
            index: index + 1,
            angle_radians: angle,
            viewer_x_m,
            viewer_y_m: electrodes.top_offset_m,
            viewer_z_m,
            cad_x_mm: viewer_x_m * MM_PER_M,
            cad_y_mm: viewer_z_m * MM_PER_M,
            cad_z_mm,
            diameter_mm: electrodes.diameter_mm,
            current_a: current_per_electrode,
            current_length_mm: electrodes.current_length_mm,
            worn_length_mm: electrodes.worn_length_mm,
            effective_length_mm,
        });
    }
    placements
}

#[pyfunction]
pub fn build_default_electrode_advisor_layout() -> ElectrodeAdvisorLayout {
    let bath = BathDefaults::new("rectangular".to_string(), 3.0, 2.0, 2.5, 1.5);
    let electrodes = ElectrodeDefaults::new(
        "graphite_standard".to_string(),
        3,
        0.1,
        1500.0,
        150.0,
        150.0,
        2500.0,
        1500.0,
    );
    let drafting = DraftingEnvelope::new(25.0, 10.0, 100.0, 2.0, 20.0);
    let placements = build_default_placements(&bath, &electrodes);

    ElectrodeAdvisorLayout {
        bath,
        electrodes,
        drafting,
        placements,
    }
}

#[pyclass(module = "tools_core.electrode_advisor")]
#[derive(Clone, Debug)]
pub struct ElectrodeAdvancementCalculator {
    #[pyo3(get)]
    pub consumption_rate: f64,
}

#[pymethods]
impl ElectrodeAdvancementCalculator {
    #[new]
    #[pyo3(signature = (consumption_rate=0.5))]
    pub fn new(consumption_rate: f64) -> PyResult<Self> {
        if consumption_rate <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "consumption_rate must be positive",
            ));
        }
        Ok(Self { consumption_rate })
    }

    pub fn calculate_consumption(&self, current_ka: f64, time_hrs: f64) -> PyResult<f64> {
        if current_ka < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "current_ka must be non-negative",
            ));
        }
        if time_hrs < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "time_hrs must be non-negative",
            ));
        }
        Ok(self.consumption_rate * current_ka * time_hrs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_default_placements_count_matches_electrode_count() {
        let bath = BathDefaults::new("rectangular".to_string(), 3.0, 2.0, 2.5, 1.5);
        let electrodes = ElectrodeDefaults::new(
            "graphite_standard".to_string(),
            3,
            0.1,
            1500.0,
            150.0,
            150.0,
            2500.0,
            1500.0,
        );
        let placements = build_default_placements(&bath, &electrodes);
        assert_eq!(placements.len(), 3);
        // Indices are 1-based and sequential.
        assert_eq!(placements[0].index, 1);
        assert_eq!(placements[2].index, 3);
    }

    #[test]
    fn placements_split_current_evenly() {
        let bath = BathDefaults::new("rectangular".to_string(), 3.0, 2.0, 2.5, 1.5);
        let electrodes = ElectrodeDefaults::new(
            "graphite_standard".to_string(),
            3,
            0.1,
            1500.0,
            150.0,
            150.0,
            3000.0,
            1500.0,
        );
        let placements = build_default_placements(&bath, &electrodes);
        for p in &placements {
            assert!(
                (p.current_a - 1000.0).abs() < 1e-9,
                "current_a={}",
                p.current_a
            );
        }
        // Effective length = current - worn.
        assert!((placements[0].effective_length_mm - 1350.0).abs() < 1e-9);
    }

    #[test]
    fn single_electrode_is_centered() {
        let bath = BathDefaults::new("rectangular".to_string(), 3.0, 2.0, 2.5, 1.5);
        let electrodes = ElectrodeDefaults::new(
            "graphite_standard".to_string(),
            1,
            0.1,
            1500.0,
            150.0,
            150.0,
            2500.0,
            1500.0,
        );
        let placements = build_default_placements(&bath, &electrodes);
        assert_eq!(placements.len(), 1);
        assert_eq!(placements[0].viewer_x_m, 0.0);
        assert_eq!(placements[0].viewer_z_m, 0.0);
    }

    #[test]
    fn default_layout_has_three_electrodes_and_mm_getters() {
        let layout = build_default_electrode_advisor_layout();
        assert_eq!(layout.placements.len(), 3);
        // Getters convert metres → millimetres.
        assert!((layout.bath_width_mm() - 3000.0).abs() < 1e-9);
        assert!((layout.glass_level_mm() - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn advancement_calculator_rejects_nonpositive_rate() {
        assert!(ElectrodeAdvancementCalculator::new(0.0).is_err());
        assert!(ElectrodeAdvancementCalculator::new(-1.0).is_err());
        assert!(ElectrodeAdvancementCalculator::new(0.5).is_ok());
    }

    #[test]
    fn advancement_calculator_consumption_is_linear() {
        let calc = ElectrodeAdvancementCalculator::new(0.5).unwrap();
        // rate * current * time
        let c = calc.calculate_consumption(10.0, 2.0).unwrap();
        assert!((c - 10.0).abs() < 1e-9, "consumption={c}");
        // Rejects negative inputs.
        assert!(calc.calculate_consumption(-1.0, 1.0).is_err());
        assert!(calc.calculate_consumption(1.0, -1.0).is_err());
    }
}

pub mod py_bindings {
    use super::*;

    pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<BathDefaults>()?;
        m.add_class::<ElectrodeDefaults>()?;
        m.add_class::<DraftingEnvelope>()?;
        m.add_class::<ElectrodePlacement>()?;
        m.add_class::<ElectrodeAdvisorLayout>()?;
        m.add_class::<ElectrodeAdvancementCalculator>()?;
        m.add_function(wrap_pyfunction!(build_default_placements, m)?)?;
        m.add_function(wrap_pyfunction!(build_default_electrode_advisor_layout, m)?)?;
        Ok(())
    }
}
