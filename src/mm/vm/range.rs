// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2022-2023 SUSE LLC
//
// Author: Joerg Roedel <jroedel@suse.de>

use crate::address::{Address, VirtAddr};
use crate::cpu::flush_tlb_global_sync;
use crate::error::SvsmError;
use crate::locking::RWLock;
use crate::mm::pagetable::{PTEntryFlags, PageTable, PageTablePart, PageTableRef};
use crate::types::{PAGE_SHIFT, PAGE_SIZE, PAGE_SIZE_2M};
use crate::utils::{align_down, align_up};

use core::cmp::max;

use intrusive_collections::rbtree::{CursorMut, RBTree};
use intrusive_collections::Bound;

use super::{Mapping, VMMAdapter, VMM};

extern crate alloc;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

/// Granularity of ranges mapped by [`struct VMR`]. The mapped region of a
/// [`struct VMR`] is always a multiple of this constant.
/// One [`VMR_GRANULE`] covers one top-level page-table entry on x86-64 with
/// 4-level paging.
pub const VMR_GRANULE: usize = PAGE_SIZE * 512 * 512 * 512;

/// Virtual Memory Region
///
/// This struct manages the mappings in a region of the virtual address space.
/// The region size is a multiple of 512GiB so that every region will fully
/// allocate one or more top-level page-table entries on x86-64. For the same
/// reason the start address must also be aligned to 512GB.
#[derive(Debug)]
pub struct VMR {
    /// Start address of this range as virtual PFN (VirtAddr >> PAGE_SHIFT).
    /// Virtual address must be aligned to [`VMR_GRANULE`] (512GB on x86-64).
    start_pfn: usize,

    /// End address of this range as virtual PFN (VirtAddr >> PAGE_SHIFT)
    /// Virtual address must be aligned to [`VMR_GRANULE`] (512GB on x86-64).
    end_pfn: usize,

    /// RBTree containing all [`struct VMM`] instances with valid mappings in
    /// the covered virtual address region. The [`struct VMM`]s are sorted by
    /// their start address and stored in an RBTree for faster lookup.
    tree: RWLock<RBTree<VMMAdapter>>,

    /// [`struct PageTableParts`] needed to map this VMR into a page-table.
    /// There is one [`struct PageTablePart`] per [`VMR_GRANULE`] covered by
    /// the region.
    pgtbl_parts: RWLock<Vec<PageTablePart>>,

    /// [`PTEntryFlags`] global to all mappings in this region. This is a
    /// combination of [`PTEntryFlags::GLOBAL`] and [`PTEntryFlags::USER`].
    pt_flags: PTEntryFlags,
}

impl VMR {
    /// Creates a new [`struct VMR`]
    ///
    /// # Arguments
    ///
    /// * `start` - Virtual start address for the memory region. Must be aligned to [`VMR_GRANULE`]
    /// * `end` - Virtual end address (non-inclusive) for the memory region.
    ///           Must be bigger than `start` and aligned to [`VMR_GRANULE`].
    /// * `flags` - Global [`PTEntryFlags`] to use for this [`struct VMR`].
    ///
    /// # Returns
    ///
    /// A new instance of [`struct VMR`].
    pub fn new(start: VirtAddr, end: VirtAddr, flags: PTEntryFlags) -> Self {
        // Global and User are per VMR flags
        VMR {
            start_pfn: start.pfn(),
            end_pfn: end.pfn(),
            tree: RWLock::new(RBTree::new(VMMAdapter::new())),
            pgtbl_parts: RWLock::new(Vec::new()),
            pt_flags: flags,
        }
    }

    /// Allocated all [`PageTablePart`]s needed to map this region
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, Err(SvsmError::Mem) on allocation error
    fn alloc_page_tables(&self) -> Result<(), SvsmError> {
        let count = ((self.end_pfn - self.start_pfn) << PAGE_SHIFT) / VMR_GRANULE;
        let start = VirtAddr::from(self.start_pfn << PAGE_SHIFT);
        let mut vec = self.pgtbl_parts.lock_write();

        for idx in 0..count {
            vec.push(PageTablePart::new(start + (idx * VMR_GRANULE)));
        }

        Ok(())
    }

    /// Populate [`PageTablePart`]s of the [`VMR`] into a page-table
    ///
    /// # Arguments
    ///
    /// * `pgtbl` - A [`PageTableRef`] pointing to the target page-table
    pub fn populate(&self, pgtbl: &mut PageTableRef) {
        let parts = self.pgtbl_parts.lock_read();

        for part in parts.iter() {
            pgtbl.populate_pgtbl_part(part);
        }
    }

    /// Initialize this [`VMR`] by checking the `start` and `end` values and
    /// allocating the [`PageTablePart`]s required for the mappings.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, Err(SvsmError::Mem) on allocation error
    pub fn initialize(&mut self) -> Result<(), SvsmError> {
        let start = VirtAddr::from(self.start_pfn << PAGE_SHIFT);
        let end = VirtAddr::from(self.end_pfn << PAGE_SHIFT);
        assert!(start < end && start.is_aligned(VMR_GRANULE) && end.is_aligned(VMR_GRANULE));

        self.alloc_page_tables()
    }

    /// Returns the virtual start and end addresses for this region
    ///
    /// # Returns
    ///
    /// Tuple containing `start` and `end` virtual address of the memory region
    fn virt_range(&self) -> (VirtAddr, VirtAddr) {
        (
            VirtAddr::from(self.start_pfn << PAGE_SHIFT),
            VirtAddr::from(self.end_pfn << PAGE_SHIFT),
        )
    }

    /// Map a [`VMM`] into the [`PageTablePart`]s of this region
    ///
    /// # Arguments
    ///
    /// - `vmm` - Reference to a [`VMM`] instance to map into the page-table
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, Err(SvsmError::Mem) on allocation error
    fn map_vmm(&self, vmm: &VMM) -> Result<(), SvsmError> {
        let (rstart, _) = self.virt_range();
        let (vmm_start, vmm_end) = vmm.range();
        let mut pgtbl_parts = self.pgtbl_parts.lock_write();
        let mapping = vmm.get_mapping();
        let pt_flags = self.pt_flags | mapping.pt_flags() | PTEntryFlags::PRESENT;
        let mut offset: usize = 0;
        let page_size = mapping.page_size();
        let shared = mapping.shared();

        while vmm_start + offset < vmm_end {
            let idx = PageTable::index::<3>(VirtAddr::from(vmm_start - rstart));
            if let Some(paddr) = mapping.map(offset) {
                if page_size == PAGE_SIZE {
                    pgtbl_parts[idx].map_4k(vmm_start + offset, paddr, pt_flags, shared)?;
                } else if page_size == PAGE_SIZE_2M {
                    pgtbl_parts[idx].map_2m(vmm_start + offset, paddr, pt_flags, shared)?;
                }
            }
            offset += page_size;
        }

        Ok(())
    }

    /// Unmap a [`VMM`] from the [`PageTablePart`]s of this region
    ///
    /// # Arguments
    ///
    /// - `vmm` - Reference to a [`VMM`] instance to unmap from the page-table
    fn unmap_vmm(&self, vmm: &VMM) {
        let (rstart, _) = self.virt_range();
        let (vmm_start, vmm_end) = vmm.range();
        let mut pgtbl_parts = self.pgtbl_parts.lock_write();
        let mapping = vmm.get_mapping();
        let page_size = mapping.page_size();
        let mut offset: usize = 0;

        while vmm_start + offset < vmm_end {
            let idx = PageTable::index::<3>(VirtAddr::from(vmm_start - rstart));
            let result = if page_size == PAGE_SIZE {
                pgtbl_parts[idx].unmap_4k(vmm_start + offset)
            } else {
                pgtbl_parts[idx].unmap_2m(vmm_start + offset)
            };

            if result.is_some() {
                mapping.unmap(offset);
            }

            offset += page_size;
        }
    }

    fn do_insert(
        &self,
        mapping: Arc<Mapping>,
        start_pfn: usize,
        cursor: &mut CursorMut<VMMAdapter>,
    ) -> Result<(), SvsmError> {
        let vmm = Box::new(VMM::new(start_pfn, mapping));
        if let Err(e) = self.map_vmm(&vmm) {
            self.unmap_vmm(&vmm);
            Err(e)
        } else {
            cursor.insert_before(vmm);
            Ok(())
        }
    }

    /// Inserts [`VMM`] at a specified virtual base address. This method
    /// checks that the [`VMM`] does not overlap with any other region.
    ///
    /// # Arguments
    ///
    /// * `vaddr` - Virtual base address to map the [`VMM`] at
    /// * `mapping` - `Rc` pointer to the VMM to insert
    ///
    /// # Returns
    ///
    /// Base address where the [`VMM`] was inserted on success or SvsmError::Mem on error
    pub fn insert_at(&self, vaddr: VirtAddr, mapping: Arc<Mapping>) -> Result<VirtAddr, SvsmError> {
        // mapping-size needs to be page-aligned
        let size = mapping.get().mapping_size() >> PAGE_SHIFT;
        let start_pfn = vaddr.pfn();
        let mut tree = self.tree.lock_write();
        let mut cursor = tree.upper_bound_mut(Bound::Included(&start_pfn));
        let mut start = self.start_pfn;
        let mut end = self.end_pfn;

        if cursor.is_null() {
            cursor = tree.front_mut();
        } else {
            let (_, node_end) = cursor.get().unwrap().range_pfn();
            start = node_end;
            cursor.move_next();
        }

        if let Some(node) = cursor.get() {
            let (node_start, _) = node.range_pfn();
            end = node_start;
        }

        let end_pfn = start_pfn + size;

        if start_pfn >= start && end_pfn <= end {
            self.do_insert(mapping, start_pfn, &mut cursor)?;
            Ok(vaddr)
        } else {
            Err(SvsmError::Mem)
        }
    }

    /// Inserts [`VMM`] with the specified alignment. This method walks the
    /// RBTree to search for a suitable region.
    ///
    /// # Arguments
    ///
    /// * `mapping` - `Rc` pointer to the VMM to insert
    /// * `align` - Alignment to use for tha mapping
    ///
    /// # Returns
    ///
    /// Base address where the [`VMM`] was inserted on success or SvsmError::Mem on error
    pub fn insert_aligned(
        &self,
        mapping: Arc<Mapping>,
        align: usize,
    ) -> Result<VirtAddr, SvsmError> {
        assert!(align.is_power_of_two());

        let size = mapping
            .get()
            .mapping_size()
            .checked_next_power_of_two()
            .unwrap_or(0)
            >> PAGE_SHIFT;
        let align = align >> PAGE_SHIFT;

        let mut start = align_up(self.start_pfn, align);
        let mut end = start;

        if size == 0 {
            return Err(SvsmError::Mem);
        }

        let mut tree = self.tree.lock_write();
        let mut cursor = tree.front_mut();

        while let Some(node) = cursor.get() {
            let (node_start, node_end) = node.range_pfn();
            end = node_start;
            if end > start && end - start >= size {
                break;
            }

            start = max(start, align_up(node_end, align));
            cursor.move_next();
        }

        if cursor.is_null() {
            end = align_down(self.end_pfn, align);
        }

        if end > start && end - start >= size {
            self.do_insert(mapping, start, &mut cursor)?;
            Ok(VirtAddr::from(start << PAGE_SHIFT))
        } else {
            Err(SvsmError::Mem)
        }
    }

    /// Inserts [`VMM`] into the virtual memory region. This method takes the
    /// next power-of-two larger of the mapping size and uses that as the
    /// alignment for the mappings base address. With that is calls
    /// [`VMR::insert_aligned`].
    ///
    /// # Arguments
    ///
    /// * `mapping` - `Rc` pointer to the VMM to insert
    ///
    /// # Returns
    ///
    /// Base address where the [`VMM`] was inserted on success or SvsmError::Mem on error
    pub fn insert(&self, mapping: Arc<Mapping>) -> Result<VirtAddr, SvsmError> {
        let align = mapping.get().mapping_size().next_power_of_two();
        self.insert_aligned(mapping, align)
    }

    /// Removes the mapping from a given base address from the RBTree
    ///
    /// # Arguments
    ///
    /// * `base` - Virtual base address of the [`VMM`] to remove
    ///
    /// # Returns
    ///
    /// The removed mapping on success, SvsmError::Mem on error
    pub fn remove(&self, base: VirtAddr) -> Result<Box<VMM>, SvsmError> {
        let mut tree = self.tree.lock_write();
        let addr = base.pfn();

        let mut cursor = tree.find_mut(&addr);
        if let Some(node) = cursor.get() {
            self.unmap_vmm(node);
            flush_tlb_global_sync();
        }
        cursor.remove().ok_or(SvsmError::Mem)
    }

    /// Dump all [`VMM`] mappings in the RBTree. This function is included for
    /// debugging purposes. And should not be called in production code.
    pub fn dump_ranges(&self) {
        let tree = self.tree.lock_read();
        for elem in tree.iter() {
            let (start_pfn, end_pfn) = elem.range_pfn();
            log::info!(
                "VMRange {:#018x}-{:#018x}",
                start_pfn << PAGE_SHIFT,
                end_pfn << PAGE_SHIFT
            );
        }
    }
}
