use std::borrow::Borrow;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use libbladerf_sys as sys;

use crate::BladeRF;
use crate::BladeRf1;
use crate::BladeRf2;
use crate::BladeRfAny;
use crate::Channel;
use crate::ChannelLayoutTx;
use crate::Result;
use crate::SampleFormat;
use crate::TxChannel;

use super::StreamConfig;

pub struct TxSyncStream<T: Borrow<D>, F: SampleFormat, D: BladeRF> {
    pub(crate) dev: T,
    pub(crate) layout: ChannelLayoutTx,
    pub(crate) config: StreamConfig,
    pub(crate) _devtype: PhantomData<D>,
    pub(crate) _format: PhantomData<F>,
}

impl<T: Borrow<D>, F: SampleFormat, D: BladeRF> TxSyncStream<T, F, D> {
    pub fn write(&self, buffer: &[F], timeout: Duration) -> Result<()> {
        let res = unsafe {
            sys::bladerf_sync_tx(
                self.dev.borrow().get_device_ptr(),
                buffer.as_ptr() as *const _,
                buffer.len() as u32,
                std::ptr::null_mut(),
                timeout.as_millis() as u32,
            )
        };
        check_res!(res);
        Ok(())
    }

    /// # Safety
    /// Need to ensure multiple streamers are not configured since a reconfiguration of one can change the sample type leading to our of bounds memory accesses.
    pub(crate) unsafe fn new(
        dev: T,
        config: StreamConfig,
        layout: ChannelLayoutTx,
    ) -> Result<TxSyncStream<T, F, D>> {
        unsafe {
            dev.borrow().set_sync_config::<F>(&config, layout.into())?;
        }

        Ok(TxSyncStream {
            dev,
            layout,
            config,
            _devtype: PhantomData,
            _format: PhantomData,
        })
    }
}

impl<'a, F: SampleFormat, D: BladeRF> TxSyncStream<&'a D, F, D> {
    fn reconfigure_inner<NF: SampleFormat>(
        self,
        config: StreamConfig,
        layout: ChannelLayoutTx,
    ) -> Result<TxSyncStream<&'a D, NF, D>> {
        // Safety: the previous streamer is moved, and is dropped so we are save to construct a new one.
        unsafe { TxSyncStream::new(self.dev, config, layout) }
    }
}

impl<F: SampleFormat, D: BladeRF> TxSyncStream<Arc<D>, F, D> {
    fn reconfigure_inner<NF: SampleFormat>(
        self,
        config: StreamConfig,
        layout: ChannelLayoutTx,
    ) -> Result<TxSyncStream<Arc<D>, NF, D>> {
        // Safety: the previous streamer is moved, and is dropped so we are save to construct a new one.
        unsafe { TxSyncStream::new(self.dev.clone(), config, layout) }
    }
}

impl<T: Borrow<D>, F: SampleFormat, D: BladeRF> Drop for TxSyncStream<T, F, D> {
    fn drop(&mut self) {
        // Ignore the results, just try disable both channels even if they don't exist on the dev.
        let _ = self.dev.borrow().set_enable_module(Channel::Tx0, false);
        let _ = self.dev.borrow().set_enable_module(Channel::Tx1, false);
    }
}

////////////////////////////////////////////////////////////////////////////////
// RX Stream Brf1

impl<T: Borrow<BladeRf1>, F: SampleFormat> TxSyncStream<T, F, BladeRf1> {
    pub fn enable(&self) -> Result<()> {
        // Safety, should be find to do a reconfigure here, nothing changes about the config, we just need to do this because disable will uninitialize the config
        unsafe {
            self.dev
                .borrow()
                .set_sync_config::<F>(&self.config, self.layout.into())?;
        }
        self.dev.borrow().set_enable_module(Channel::Tx0, true)
    }

    pub fn disable(&self) -> Result<()> {
        self.dev.borrow().set_enable_module(Channel::Tx0, false)
    }
}

impl<'a, F: SampleFormat> TxSyncStream<&'a BladeRf1, F, BladeRf1> {
    pub fn reconfigure<NF: SampleFormat>(
        self,
        config: StreamConfig,
    ) -> Result<TxSyncStream<&'a BladeRf1, NF, BladeRf1>> {
        self.reconfigure_inner(config, ChannelLayoutTx::SISO(TxChannel::Tx0))
    }
}

impl<F: SampleFormat> TxSyncStream<Arc<BladeRf1>, F, BladeRf1> {
    pub fn reconfigure<NF: SampleFormat>(
        self,
        config: StreamConfig,
    ) -> Result<TxSyncStream<Arc<BladeRf1>, NF, BladeRf1>> {
        self.reconfigure_inner(config, ChannelLayoutTx::SISO(TxChannel::Tx0))
    }
}

////////////////////////////////////////////////////////////////////////////////
// RX Stream Brf2

impl<T: Borrow<BladeRf2> + Clone, F: SampleFormat> TxSyncStream<T, F, BladeRf2> {
    pub fn enable(&self) -> Result<()> {
        // Safety, should be find to do a reconfigure here, nothing changes about the config, we just need to do this because disable will uninitialize the config
        unsafe {
            self.dev
                .borrow()
                .set_sync_config::<F>(&self.config, self.layout.into())?;
        }
        match self.layout {
            ChannelLayoutTx::SISO(ch) => self.dev.borrow().set_enable_module(ch.into(), true),
            ChannelLayoutTx::MIMO => {
                self.dev.borrow().set_enable_module(Channel::Tx0, true)?;
                self.dev.borrow().set_enable_module(Channel::Tx1, true)?;
                Ok(())
            }
        }
    }

    pub fn disable(&self) -> Result<()> {
        match self.layout {
            ChannelLayoutTx::SISO(ch) => self.dev.borrow().set_enable_module(ch.into(), false),
            ChannelLayoutTx::MIMO => {
                self.dev.borrow().set_enable_module(Channel::Tx0, false)?;
                self.dev.borrow().set_enable_module(Channel::Tx1, false)?;
                Ok(())
            }
        }
    }
}

impl<'a, F: SampleFormat> TxSyncStream<&'a BladeRf2, F, BladeRf2> {
    pub fn reconfigure<NF: SampleFormat>(
        self,
        config: StreamConfig,
        layout: ChannelLayoutTx,
    ) -> Result<TxSyncStream<&'a BladeRf2, NF, BladeRf2>> {
        self.reconfigure_inner(config, layout)
    }
}

impl<F: SampleFormat> TxSyncStream<Arc<BladeRf2>, F, BladeRf2> {
    pub fn reconfigure<NF: SampleFormat>(
        self,
        config: StreamConfig,
        layout: ChannelLayoutTx,
    ) -> Result<TxSyncStream<Arc<BladeRf2>, NF, BladeRf2>> {
        self.reconfigure_inner(config, layout)
    }
}

////////////////////////////////////////////////////////////////////////////////
// RX Stream BrfAny

impl<T: Borrow<BladeRfAny> + Clone, F: SampleFormat> TxSyncStream<T, F, BladeRfAny> {
    pub fn enable(&self) -> Result<()> {
        // Safety, should be find to do a reconfigure here, nothing changes about the config, we just need to do this because disable will uninitialize the config
        unsafe {
            self.dev
                .borrow()
                .set_sync_config::<F>(&self.config, self.layout.into())?;
        }
        match self.layout {
            ChannelLayoutTx::SISO(ch) => self.dev.borrow().set_enable_module(ch.into(), true),
            ChannelLayoutTx::MIMO => {
                self.dev.borrow().set_enable_module(Channel::Tx0, true)?;
                self.dev.borrow().set_enable_module(Channel::Tx1, true)?;
                Ok(())
            }
        }
    }

    pub fn disable(&self) -> Result<()> {
        match self.layout {
            ChannelLayoutTx::SISO(ch) => self.dev.borrow().set_enable_module(ch.into(), false),
            ChannelLayoutTx::MIMO => {
                self.dev.borrow().set_enable_module(Channel::Tx0, false)?;
                self.dev.borrow().set_enable_module(Channel::Tx1, false)?;
                Ok(())
            }
        }
    }
}

impl<'a, F: SampleFormat> TxSyncStream<&'a BladeRfAny, F, BladeRfAny> {
    pub fn reconfigure<NF: SampleFormat>(
        self,
        config: StreamConfig,
        layout: ChannelLayoutTx,
    ) -> Result<TxSyncStream<&'a BladeRfAny, NF, BladeRfAny>> {
        self.reconfigure_inner(config, layout)
    }
}

impl<F: SampleFormat> TxSyncStream<Arc<BladeRfAny>, F, BladeRfAny> {
    pub fn reconfigure<NF: SampleFormat>(
        self,
        config: StreamConfig,
        layout: ChannelLayoutTx,
    ) -> Result<TxSyncStream<Arc<BladeRfAny>, NF, BladeRfAny>> {
        self.reconfigure_inner(config, layout)
    }
}
