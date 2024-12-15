use std::{any::Any, error::Error, io, rc::Rc, str::FromStr};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Stylize,
    symbols::border,
    text::{Line, Text},
    widgets::{Block, Borders, List, Paragraph, Widget},
    DefaultTerminal, Frame,
};

use ratatui::prelude::*;

use bladerf::{
    BladeRF, Correction, CorrectionDcOffsetI, CorrectionDcOffsetQ, CorrectionGain, CorrectionPhase,
    CorrectionValue,
};
use tui_textarea::{Input, Key, TextArea};

#[derive(Debug, Clone, Copy)]
enum SelectedInput {
    Frequency,
    DcOffsetI,
    DcOffsetQ,
    Phase,
    Gain,
}

impl SelectedInput {
    fn up(&mut self) {
        *self = match self {
            SelectedInput::Frequency => SelectedInput::Gain,
            SelectedInput::DcOffsetI => SelectedInput::Frequency,
            SelectedInput::DcOffsetQ => SelectedInput::DcOffsetI,
            SelectedInput::Phase => SelectedInput::DcOffsetQ,
            SelectedInput::Gain => SelectedInput::Phase,
        }
    }
    fn down(&mut self) {
        *self = match self {
            SelectedInput::Frequency => SelectedInput::DcOffsetI,
            SelectedInput::DcOffsetI => SelectedInput::DcOffsetQ,
            SelectedInput::DcOffsetQ => SelectedInput::Phase,
            SelectedInput::Phase => SelectedInput::Gain,
            SelectedInput::Gain => SelectedInput::Frequency,
        }
    }
}

pub struct App {
    channel: bladerf::Channel,
    device: BladeRF,
    selected_input: SelectedInput,
    focused: bool,
    exit: bool,
}

type IntValidationFunction<T, E> = Box<dyn Fn(&str) -> Result<T, E>>;

fn validate_frequency(val: &str) -> Result<u64, String> {
    match val.parse::<u64>() {
        Err(err) => Err(format!("{}", err)),
        Ok(freq) if (freq > 300000000) && (freq < 3000000000) => Ok(freq),
        Ok(invalid_freq) => Err(format!("Value `{}` out of range", invalid_freq)),
    }
}

fn validate_correction<T: CorrectionValue>(val: &str) -> Result<T, String> {
    match val.parse::<i16>().map(|x| T::new(x)) {
        Err(err) => Err(format!("{}", err)),
        Ok(Some(x)) => Ok(x),
        Ok(None) => Err(format!("Value `{val}` out of range")),
    }
}

/// A custom numeric input widget with validation
pub struct NumericInput<'a, T, E> {
    textarea: TextArea<'a>,
    validation_fn: IntValidationFunction<T, E>, // Validation logic
}

impl<'a, T> NumericInput<'a, T, String> {
    /// Creates a new `NumericInput` with the provided initial value and validation function.
    pub fn new<F>(initial_value: String, validation_fn: F) -> Self
    where
        F: Fn(&str) -> Result<T, String> + 'static,
    {
        let mut numeric_input = Self {
            textarea: TextArea::new(vec![initial_value]),
            validation_fn: Box::new(validation_fn),
        };
        numeric_input.validate();
        numeric_input.remove_focus_inner();
        numeric_input
    }

    fn validate(&mut self) {
        match (self.validation_fn)(&self.textarea.lines()[0]) {
            Ok(_) => {
                self.textarea
                    .set_style(Style::default().fg(Color::LightGreen));
                self.textarea.set_block(
                    Block::default()
                        .border_style(Color::LightGreen)
                        .borders(Borders::ALL)
                        .title("OK"),
                );
            }
            Err(err) => {
                self.textarea
                    .set_style(Style::default().fg(Color::LightRed));
                self.textarea.set_block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Color::LightRed)
                        .title(format!("ERROR: {err}")),
                );
            }
        }
    }
    /// Handles input events and revalidates the value
    pub fn handle_input_inner(&mut self, input: Input) {
        if self.textarea.input(input) {
            self.validate();
        }
    }

    /// Sets focus (cursor style) to this input
    pub fn set_focus_inner(&mut self) {
        self.textarea
            .set_cursor_style(Style::default().add_modifier(Modifier::REVERSED));
    }

    /// Removes focus from this input
    pub fn remove_focus_inner(&mut self) {
        self.textarea.set_cursor_style(Style::default());
    }

    /// Retrieves the current value as a string
    pub fn value(&self) -> String {
        self.textarea.lines().join("")
    }
}

trait NumericInputHandle {
    fn handle_input(&mut self, input: Input);
    fn set_focus(&mut self);
    fn unset_focus(&mut self);
    fn num_render(&self, area: Rect, buf: &mut Buffer);
}

impl<'a, T> NumericInputHandle for &mut NumericInput<'a, T, String> {
    fn handle_input(&mut self, input: Input) {
        self.handle_input_inner(input);
    }

    fn set_focus(&mut self) {
        self.set_focus_inner();
    }

    fn unset_focus(&mut self) {
        self.remove_focus_inner();
    }

    fn num_render(&self, area: Rect, buf: &mut Buffer) {
        self.textarea.render(area, buf);
    }
}

impl<'a, T> NumericInputHandle for NumericInput<'a, T, String> {
    fn handle_input(&mut self, input: Input) {
        self.handle_input_inner(input);
    }

    fn set_focus(&mut self) {
        self.set_focus_inner();
    }

    fn unset_focus(&mut self) {
        self.remove_focus_inner();
    }

    fn num_render(&self, area: Rect, buf: &mut Buffer) {
        self.textarea.render(area, buf);
    }
}

impl<'a, T, E> Widget for &NumericInput<'a, T, E> {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        self.textarea.render(area, buf);
    }
}

impl<'a, T, E> Widget for NumericInput<'a, T, E> {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        self.textarea.render(area, buf);
    }
}

trait NumericInputWidget: NumericInputHandle + Widget {}

impl<'a, T> NumericInputWidget for NumericInput<'a, T, String> {}

impl Widget for &dyn NumericInputWidget {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        self.num_render(area, buf);
    }
}

impl Widget for Box<dyn NumericInputWidget> {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        self.num_render(area, buf);
    }
}

trait BoxWidget {
    fn render_box(self: Box<Self>, area: Rect, buf: &mut Buffer);
}

impl<W: Widget> BoxWidget for W {
    fn render_box(self: Box<Self>, area: Rect, buf: &mut Buffer) {
        (*self).render(area, buf)
    }
}

// impl<W: BoxWidget + ?Sized> Widget for Box<W> {
//     fn render(self, area: Rect, buf: &mut Buffer) {
//         self.render_box(area, buf)
//     }
// }

impl App {
    fn new(dev: BladeRF) -> App {
        let channel = bladerf::Channel::Tx1;
        App {
            channel,
            device: dev,
            selected_input: SelectedInput::Frequency,
            focused: false,
            exit: false,
        }
    }

    /// runs the application's main loop until the user quits
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        let mut frequency_input =
            NumericInput::new(self.get_freq().to_string(), validate_frequency);

        let mut icorr_input = NumericInput::new(self.get_icorr().to_string(), |x| {
            validate_correction::<CorrectionDcOffsetI>(x)
        });

        let mut qcorr_input = NumericInput::new(self.get_qcorr().to_string(), |x| {
            validate_correction::<CorrectionDcOffsetQ>(x)
        });

        let mut phase_input = NumericInput::new(self.get_phase().to_string(), |x| {
            validate_correction::<CorrectionPhase>(x)
        });

        let mut gain_input = NumericInput::new(self.get_gain().to_string(), |x| {
            validate_correction::<CorrectionGain>(x)
        });

        while !self.exit {
            let debug_test = Text::from(format!("Sel: {:?}", self.selected_input));

            frequency_input.unset_focus();
            icorr_input.unset_focus();
            qcorr_input.unset_focus();
            phase_input.unset_focus();
            gain_input.unset_focus();

            let current_setpoint = vec![
                Paragraph::new(self.get_freq().to_string())
                    .block(Block::new().borders(Borders::ALL).title("Set Frequency")),
                Paragraph::new(self.get_icorr().to_string())
                    .block(Block::new().borders(Borders::ALL).title("Set ICorr")),
                Paragraph::new(self.get_qcorr().to_string())
                    .block(Block::new().borders(Borders::ALL).title("Set QCorr")),
                Paragraph::new(self.get_phase().to_string())
                    .block(Block::new().borders(Borders::ALL).title("Set Phase")),
                Paragraph::new(self.get_gain().to_string())
                    .block(Block::new().borders(Borders::ALL).title("Set Gain")),
            ];

            if self.focused {
                match self.selected_input {
                    SelectedInput::Frequency => frequency_input.set_focus(),
                    SelectedInput::DcOffsetI => icorr_input.set_focus(),
                    SelectedInput::DcOffsetQ => qcorr_input.set_focus(),
                    SelectedInput::Phase => phase_input.set_focus(),
                    SelectedInput::Gain => gain_input.set_focus(),
                }
            }

            let selected_idx = match self.selected_input {
                SelectedInput::Frequency => 0_usize,
                SelectedInput::DcOffsetI => 1,
                SelectedInput::DcOffsetQ => 2,
                SelectedInput::Phase => 3,
                SelectedInput::Gain => 4,
            };

            terminal.draw(|frame| {
                let row_layout = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints(vec![
                        Constraint::Length(3),
                        Constraint::Length(3),
                        Constraint::Length(3),
                        Constraint::Length(3),
                        Constraint::Length(3),
                        Constraint::Length(3),
                    ])
                    .split(frame.area());

                let column_layout: Vec<Rc<[Rect]>> = row_layout
                    .iter()
                    .map(|layout| {
                        Layout::default()
                            .direction(Direction::Horizontal)
                            .constraints(vec![
                                Constraint::Length(1),
                                Constraint::Percentage(50),
                                Constraint::Percentage(50),
                            ])
                            .split(*layout)
                    })
                    .collect();

                frame.render_widget(&frequency_input, column_layout[0][1]);
                frame.render_widget(&icorr_input, column_layout[1][1]);
                frame.render_widget(&qcorr_input, column_layout[2][1]);
                frame.render_widget(&phase_input, column_layout[3][1]);
                frame.render_widget(&gain_input, column_layout[4][1]);

                for (idx, (layout, setpoint)) in
                    column_layout.iter().zip(current_setpoint).enumerate()
                {
                    if idx == selected_idx {
                        frame.render_widget(Text::from(vec![" ".into(), ">".into()]), layout[0]);
                    } else {
                        frame.render_widget(" ", layout[0]);
                    }
                    frame.render_widget(setpoint, layout[2]);
                }

                frame.render_widget(debug_test, row_layout[5]);
            })?;

            let update_corrs = if self.focused {
                match self.selected_input {
                    SelectedInput::Frequency => self.handle_events(Some(&mut frequency_input))?,
                    SelectedInput::DcOffsetI => self.handle_events(Some(&mut icorr_input))?,
                    SelectedInput::DcOffsetQ => self.handle_events(Some(&mut qcorr_input))?,
                    SelectedInput::Phase => self.handle_events(Some(&mut phase_input))?,
                    SelectedInput::Gain => self.handle_events(Some(&mut gain_input))?,
                }
            } else {
                self.handle_events::<u8>(None)?
            };

            if update_corrs {
                if let Ok(val) = (frequency_input.validation_fn)(frequency_input.value().as_str()) {
                    self.set_freq(val);
                }
                if let Ok(val) = (icorr_input.validation_fn)(icorr_input.value().as_str()) {
                    self.set_corr(val);
                }
                if let Ok(val) = (qcorr_input.validation_fn)(qcorr_input.value().as_str()) {
                    self.set_corr(val);
                }
                if let Ok(val) = (phase_input.validation_fn)(phase_input.value().as_str()) {
                    self.set_corr(val);
                }
                if let Ok(val) = (gain_input.validation_fn)(gain_input.value().as_str()) {
                    self.set_corr(val);
                }
            }
        }
        Ok(())
    }

    fn selected_up(&mut self) {
        self.selected_input.up();
    }

    fn selected_down(&mut self) {
        self.selected_input.down();
    }

    fn exit(&mut self) {
        self.exit = true;
    }

    fn set_focus(&mut self) {
        self.focused = true;
    }

    fn unset_focus(&mut self) {
        self.focused = false;
    }

    fn get_freq(&self) -> u64 {
        self.device.get_frequency(self.channel).unwrap()
    }

    fn get_icorr(&self) -> i16 {
        self.device
            .get_correction::<CorrectionDcOffsetI>(self.channel)
            .unwrap()
            .into_inner()
    }

    fn get_qcorr(&self) -> i16 {
        self.device
            .get_correction::<CorrectionDcOffsetQ>(self.channel)
            .unwrap()
            .into_inner()
    }

    fn get_phase(&self) -> i16 {
        self.device
            .get_correction::<CorrectionPhase>(self.channel)
            .unwrap()
            .into_inner()
    }

    fn get_gain(&self) -> i16 {
        self.device
            .get_correction::<CorrectionGain>(self.channel)
            .unwrap()
            .into_inner()
    }

    fn set_freq(&self, freq: u64) {
        self.device.set_frequency(self.channel, freq).unwrap()
    }

    fn set_corr<T: CorrectionValue>(&self, corr: T) {
        self.device.set_correction(self.channel, corr).unwrap()
    }

    /// updates the application's state based on user input
    fn handle_events<T>(
        &mut self,
        idk: Option<&mut NumericInput<'_, T, String>>,
    ) -> io::Result<bool> {
        let mut need_to_update = false;
        if let Some(idk2) = idk {
            match crossterm::event::read()?.into() {
                Input { key: Key::Esc, .. } => self.exit(),
                // Input { key: Key::Up, .. } => self.selected_up(),
                // Input { key: Key::Down, .. } => self.selected_down(),
                Input {
                    key: Key::Enter, ..
                } => {
                    need_to_update = true;
                    self.unset_focus();
                }

                input => idk2.handle_input(input),
            }
        } else {
            match crossterm::event::read()?.into() {
                Input { key: Key::Esc, .. } => self.exit(),
                Input { key: Key::Up, .. } => self.selected_up(),
                Input { key: Key::Down, .. } => self.selected_down(),
                Input {
                    key: Key::Enter, ..
                } => self.set_focus(),
                _ => {}
            }
        }

        Ok(need_to_update)
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Char('q') => self.exit(),
            KeyCode::Up => self.selected_up(),
            KeyCode::Down => self.selected_down(),
            _ => {}
        }
    }
}

impl Widget for &App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let title = Line::from(" BladeRF SigGen ".bold());

        Paragraph::new(title).render(area, buf);
    }
}

fn main() -> io::Result<()> {
    let device =
        BladeRF::open_first().map_err(|err| io::Error::new(io::ErrorKind::NotFound, err))?;

    let mut terminal = ratatui::init();
    let app_result = App::new(device).run(&mut terminal);
    ratatui::restore();
    app_result
}