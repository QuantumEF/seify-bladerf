use anyhow::Context;
use bladerf::{
    BladeRF, BladeRfAny, ChannelLayoutRx, ComplexI16, Gain, GainMode, RxChannel, StreamConfig,
};
use indicatif::{ProgressBar, ProgressStyle};
use num_complex::Complex;
use rustradio::sigmf::{Capture, SigMF};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::mpsc::TryRecvError,
    time::Duration,
};

use clap::{Parser, ValueEnum};

#[derive(ValueEnum, Clone, Copy, Debug)]
enum CliChannel {
    Ch0,
    Ch1,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum CliGainMode {
    Default,
    FastAttackAgc,
    SlowAttackAgc,
    HybridAgc,
}

const NUM_BUFFERS: u32 = 16;
const NUM_TRANSFERS: u32 = 8;
const SAMPLES_PER_BLOCK: usize = 8192;

/// AI generated function with human modification
fn check_precision_loss(val: u64) -> Option<f64> {
    let float_val = val as f64;
    let round_trip = float_val as u64;

    let lost_precision = val != round_trip;

    if lost_precision {
        None
    } else {
        Some(float_val)
    }
}

/// Simple program to receive samples from a bladeRF and write them to a file.
///
/// The output file will be a binary file containing interleaved I and Q samples
/// where each sample is a 16-bit little endian signed integer.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// The output file to write samples to.
    #[arg(short, long)]
    outfile: PathBuf,

    /// The device identifier.
    ///
    /// Valid options are described here: <https://www.nuand.com/libbladeRF-doc/v2.5.0/group___f_n___i_n_i_t.html#gab341ac98615f393da9158ea59cdb6a24>
    #[arg(short, long)]
    device: Option<String>,

    /// The center frequency to tune to in Hz.
    #[arg(short, long)]
    frequency: u64,

    /// The sample rate of the device in Hz (samples per second).
    #[arg(short, long)]
    samplerate: u32,

    /// The channel/port to use
    #[arg(short, long, default_value = "ch0")]
    channel: CliChannel,

    /// How long to recieve samples for in seconds. If not provided, will run indefinitely.
    #[arg(long, short = 't')]
    duration: Option<f32>,

    /// RX Gain
    ///
    /// Leaving unset attemps to configure AGC
    #[arg(long, short = 'g', value_parser = clap::value_parser!(i32).range(0..=60))]
    gain: Option<Gain>,

    /// Bandwidth in MHz
    #[arg(long, short)]
    bandwidth: Option<u32>,

    /// Bladerf Gain Mode
    ///
    /// Maps to the gain modes is libbladerf, see https://www.nuand.com/libbladeRF-doc/v2.5.0/group___f_n___g_a_i_n.html#gae7632e9f6b3a5a182ef012c214be0f78 for fore information.
    /// If unset, BLADERF_GAIN_DEFAULT is used.
    /// For manual gain control, simply set the `--gain` option.
    #[arg(long, default_value = "default")]
    gain_mode: CliGainMode,

    /// Disable progress bar
    #[arg(long)]
    noprogress: bool,
}

fn complex_i16_to_u8(arr: &[ComplexI16]) -> &[u8] {
    let len = std::mem::size_of_val(arr);
    let ptr = arr.as_ptr() as *const u8;
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    pretty_env_logger::init();

    log::debug!("Args: {:#?}", args);

    let dev = if let Some(device) = args.device {
        log::debug!("Opening device with device identifier: {}", device);
        BladeRfAny::open_identifier(&device).with_context(|| "Cannot Open Device")?
    } else {
        log::debug!("Opening first device");
        BladeRfAny::open_first().with_context(|| "Cannot Open Device")?
    };

    let channel = match args.channel {
        CliChannel::Ch0 => RxChannel::Rx0,
        CliChannel::Ch1 => RxChannel::Rx1,
    };

    log::debug!("Configuring channel {:?}", channel);

    dev.set_frequency(channel.into(), args.frequency)
        .with_context(|| {
            format!(
                "Unable to set frequency ({}) on the given channel ({:?}).",
                args.frequency, channel
            )
        })?;

    let get_freq = dev
        .get_frequency(channel.into())
        .with_context(|| "Unable to retrieve/sanity check the set frequency")?;
    if get_freq != args.frequency {
        log::warn!(
            "Frequency requested, {}, does not match the set frequency, {}",
            args.frequency,
            get_freq
        );
    }
    log::debug!("Frequency set to {}", get_freq);

    dev.set_sample_rate(channel.into(), args.samplerate)
        .with_context(|| {
            format!(
                "Unable to set sample rate ({}) on the given channel ({:?}).",
                args.samplerate, channel
            )
        })?;
    let get_samplerate = dev
        .get_sample_rate(channel.into())
        .with_context(|| "Unable to retrieve the sample rate for a sanity check")?;

    if get_samplerate != args.samplerate {
        log::warn!(
            "Requested sample rate, {}, does not match the sample rate set, {}",
            args.samplerate,
            get_samplerate
        );
    }

    log::debug!("Sample rate set to {}", args.samplerate);

    let (set_gain_mode, set_gain) = if let Some(gain) = args.gain {
        let set_gain_mode = GainMode::Manual;
        dev.set_gain_mode(channel.into(), set_gain_mode)
            .with_context(|| "Unable to set manual gain mode")?;

        dev.set_gain(channel.into(), gain)
            .with_context(|| format!("Unable to set the RX gain to {gain} dB"))?;

        (set_gain_mode, Some(gain))
    } else {
        let gain_mode = match args.gain_mode {
            CliGainMode::Default => GainMode::Default,
            CliGainMode::FastAttackAgc => GainMode::FastAttackAgc,
            CliGainMode::SlowAttackAgc => GainMode::SlowAttackAgc,
            CliGainMode::HybridAgc => GainMode::HybridAgc,
        };
        dev.set_gain_mode(channel.into(), gain_mode)
            .with_context(|| format!("Unable to set gain mode of {:?}", args.gain_mode))?;

        (gain_mode, None)
    };

    let get_gain = dev
        .get_gain(channel.into())
        .with_context(|| "Unable to get the gain for a sanity check")?;

    if let Some(gain) = set_gain {
        if get_gain != gain {
            log::warn!(
                "Gain requested, {}, does not match the set gain, {}",
                gain,
                get_gain
            );
        }
    }
    log::debug!("RX gain set to {} dB", get_gain);

    let get_gain_mode = dev
        .get_gain_mode(channel.into())
        .with_context(|| "Unable to get the gain mode for a sanity check")?;
    if get_gain_mode != set_gain_mode {
        log::warn!(
            "Gain mode requested, {:?}, does not match the set gain mode, {:?}",
            GainMode::Manual,
            get_gain_mode
        );
    }
    log::debug!("Gain mode set to {:?}", get_gain_mode);

    if let Some(bandwidth) = args.bandwidth {
        let set_bw = dev
            .set_bandwidth(channel.into(), bandwidth)
            .with_context(|| "Unable to set the bandwidth")?;

        if set_bw != bandwidth {
            log::warn!(
                "Requested bandwidth {bandwidth} is not equal to the bandwidth set {set_bw}."
            );
        }
    }

    let get_bandwidth = dev
        .get_bandwidth(channel.into())
        .with_context(|| "Unable to get the current device bandwidth")?;
    log::debug!("Device bandwidth is {get_bandwidth}.");

    let config = StreamConfig::new(
        NUM_BUFFERS,
        SAMPLES_PER_BLOCK,
        NUM_TRANSFERS,
        Duration::from_secs(3),
    )
    .with_context(|| "Cannot Create Sync Config")?;
    let layout = ChannelLayoutRx::SISO(channel);
    let reciever = dev
        .rx_streamer::<ComplexI16>(config, layout)
        .with_context(|| "Cannot Get Streamer")?;

    let meta_filename = {
        let mut name = args.outfile.clone();
        name.set_extension("sigmf-meta");
        name
    };

    let mut meta_file =
        File::create(meta_filename).with_context(|| "Cannout open output sigmf-meta file")?;

    const BLADERF_SIGMF_SAMPLE_FORMAT: &str = "ci16_le";
    let mut sigmf_meta = SigMF::new(BLADERF_SIGMF_SAMPLE_FORMAT.to_owned());

    sigmf_meta.global.core_author = None;
    sigmf_meta.global.core_hw = Some(dev.get_board_name().to_owned());
    sigmf_meta.global.core_sample_rate = Some(get_samplerate.into());
    sigmf_meta.global.core_description = Some(format!(
        "Bladerf using: channel {channel:?}, gain mode: {get_gain_mode:?}, gain: {get_gain:?}, bandwidth: {get_bandwidth}"
    ));
    let core_frequency = check_precision_loss(get_freq);
    if core_frequency.is_none() {
        log::warn!("Unable to write frequency to sigmf metadata file due to precision loss");
    };
    sigmf_meta.captures.push(Capture {
        core_sample_start: 0,
        core_global_index: None,
        core_header_bytes: None,
        core_frequency,
        core_datetime: None,
    });

    let serialized_meta = serde_json::to_string_pretty(&sigmf_meta)
        .with_context(|| "Unable to serialize metadata")?;

    meta_file
        .write_all(serialized_meta.as_bytes())
        .with_context(|| "Unable to write metadata to file")?;
    meta_file
        .flush()
        .with_context(|| "Unable to flush metadata file")?;

    let data_filename = {
        let mut name = args.outfile.clone();
        name.set_extension("sigmf-data");
        name
    };

    let data_file =
        File::create(data_filename).with_context(|| "Cannot Open Output sigmf-data File")?;
    let mut file_buf = BufWriter::new(data_file);
    let mut buffer = [Complex::new(0_i16, 0); SAMPLES_PER_BLOCK];

    log::debug!("Opened file for writing");

    reciever.enable().with_context(|| "Cannot Enable Stream")?;

    log::debug!("Stream enabled");

    let (ctrlc_tx, ctrlc_rx) = std::sync::mpsc::channel();
    ctrlc::set_handler(move || {
        let _ = ctrlc_tx.send(());
    })
    .with_context(|| "Cannot Set Ctrl-C Handler")?;

    log::info!("Starting to receive samples");

    let bar_style = ProgressStyle::with_template(
        "{spinner:.blue} [{elapsed_precise}] {binary_bytes} written to disk.",
    )
    .unwrap();
    let progress = ProgressBar::no_length().with_style(bar_style);

    let mut reciever_inner = || -> anyhow::Result<()> {
        reciever
            .read(&mut buffer, Duration::from_secs(1))
            .with_context(|| "Cannot Read Samples")?;

        let data = complex_i16_to_u8(&buffer);

        file_buf
            .write_all(data)
            .with_context(|| "Could not write to file")?;

        if !args.noprogress {
            progress.inc(SAMPLES_PER_BLOCK as u64 * size_of::<ComplexI16>() as u64);
        }

        Ok(())
    };

    match args.duration {
        Some(duration) => {
            let buffer_read_count_limit = {
                let sample_count = args.samplerate as f64 * duration as f64;
                let samples_per_block = SAMPLES_PER_BLOCK as f64;
                (sample_count / samples_per_block) as u64
            };

            for _ in 0..buffer_read_count_limit {
                reciever_inner()?;
                match ctrlc_rx.try_recv() {
                    std::result::Result::Ok(_) => break,
                    Err(TryRecvError::Disconnected) => break,
                    _ => {}
                }
            }
        }
        None => loop {
            reciever_inner()?;
            match ctrlc_rx.try_recv() {
                std::result::Result::Ok(_) => break,
                Err(TryRecvError::Disconnected) => break,
                _ => {}
            }
        },
    }

    log::info!("Finished receiving samples");

    file_buf.flush().with_context(|| "Cannot Flush File")?;
    let file = file_buf.into_inner().with_context(|| "Cannot Get File")?;
    file.sync_all().with_context(|| "Cannot Sync File")?;

    Ok(())
}
