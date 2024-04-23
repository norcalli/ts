use std::{collections::BTreeSet, ops::Index, sync::Arc};

use array_macro::array;
use crossbeam_channel as ch;
use once_cell::sync::OnceCell;
use ordered_float::NotNan;
use ustr::Ustr;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub enum Type {
    Nil,
    Atom,
    I64,
    U64,
    F64,
}

#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, derive_more::From)]
pub enum E {
    NilLower,
    // StringNil,
    // String(String),
    AtomLower,
    Atom(Ustr),
    AtomUpper,
    I64Lower,
    I64(i64),
    I64Upper,
    U64Lower,
    U64(u64),
    U64Upper,
    F64Lower,
    F64(NotNan<f64>),
    F64Upper,
    NilUpper,
}

impl std::fmt::Debug for E {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            E::NilLower => write!(f, "<nil"),
            E::NilUpper => write!(f, ">nil"),
            E::AtomLower => write!(f, "<atom"),
            E::Atom(x) => write!(f, "{x}"),
            E::AtomUpper => write!(f, ">atom"),
            E::I64Lower => write!(f, "<i64"),
            E::I64(x) => write!(f, "{x}"),
            E::I64Upper => write!(f, ">i64"),
            E::U64Lower => write!(f, "<u64"),
            E::U64(x) => write!(f, "{x}"),
            E::U64Upper => write!(f, ">u64"),
            E::F64Lower => write!(f, "<f64"),
            E::F64(x) => write!(f, "{x}"),
            E::F64Upper => write!(f, ">f64"),
        }
    }
}

impl std::fmt::Display for E {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            E::NilLower => write!(f, "<nil"),
            E::NilUpper => write!(f, ">nil"),
            E::AtomLower => write!(f, "<atom"),
            E::Atom(x) => write!(f, "{x}"),
            E::AtomUpper => write!(f, ">atom"),
            E::I64Lower => write!(f, "<i64"),
            E::I64(x) => write!(f, "{x}"),
            E::I64Upper => write!(f, ">i64"),
            E::U64Lower => write!(f, "<u64"),
            E::U64(x) => write!(f, "{x}"),
            E::U64Upper => write!(f, ">u64"),
            E::F64Lower => write!(f, "<f64"),
            E::F64(x) => write!(f, "{x}"),
            E::F64Upper => write!(f, ">f64"),
        }
    }
}

macro_rules! add_try_into {
    ($ty:ty: $variant:ident($inner:ident) => $to:expr) => {
        impl TryInto<$ty> for E {
            type Error = anyhow::Error;

            #[inline]
            fn try_into(self) -> Result<$ty, Self::Error> {
                Ok(match self {
                    Self::$variant($inner) => $to,
                    _ => anyhow::bail!("Can't coerce {self:?} to {}", stringify!($ty)),
                })
            }
        }
    };
}

add_try_into!(ustr::Ustr: Atom(atom) => atom);
add_try_into!(&'static str: Atom(atom) => atom.as_str());
add_try_into!(i64: I64(v) => v);
add_try_into!(i32: I64(v) => v.try_into()?);
add_try_into!(i16: I64(v) => v.try_into()?);
add_try_into!(i8: I64(v) => v.try_into()?);
add_try_into!(NotNan<f64>: F64(v) => v);
add_try_into!(f64: F64(v) => v.into_inner());
add_try_into!(f32: F64(v) => v.into_inner() as f32);
add_try_into!(u64: U64(v) => v);

/* impl<'a> TryInto<&'a str> for E {
    type Error = anyhow::Error;
    fn try_into(self) -> Result<&'a str, Self::Error> {
        Ok(match self {
            Self::Atom(atom) => atom.as_str(),
            _ => anyhow::bail!("Can't coerce {self:?} to str"),
        })
    }
} */

impl From<&str> for E {
    fn from(value: &str) -> Self {
        Self::Atom(ustr::ustr(value))
    }
}

impl From<f64> for E {
    fn from(value: f64) -> Self {
        Self::F64(NotNan::new(value).unwrap())
    }
}

impl From<i32> for E {
    fn from(value: i32) -> Self {
        Self::I64(value as i64)
    }
}

impl From<Type> for E {
    fn from(value: Type) -> Self {
        match value {
            Type::Nil => E::NilLower,
            Type::Atom => E::AtomLower,
            Type::I64 => E::I64Lower,
            Type::U64 => E::U64Lower,
            Type::F64 => E::F64Lower,
        }
    }
}

impl E {
    /* fn is_pattern(self) -> bool {
        match self {
            Self::AtomLower
            | Self::I64Lower
            | Self::U64Lower
            | Self::F64Lower
            | Self::AtomUpper
            | Self::I64Upper
            | Self::U64Upper
            | Self::F64Upper => true,
            _ => false,
        }
    }
    fn is_nil(self) -> bool {
        match self {
            Self::Nil => true,
            _ => false,
        }
    }

    fn type_(self) -> Type {
        match self {
            Self::Nil => Type::Nil,
            Self::AtomLower | Self::AtomUpper | Self::Atom(_) => Type::Atom,
            Self::I64Lower | Self::I64Upper | Self::I64(_) => Type::I64,
            Self::U64Lower | Self::U64Upper | Self::U64(_) => Type::U64,
            Self::F64Lower | Self::F64Upper | Self::F64(_) => Type::F64,
        }
    }

    fn types_match(self, other: Self) -> bool {
        self.type_() == other.type_()
    } */

    /* fn upper(self) -> Self {
        match self {
            Self::AtomLower => Self::AtomUpper,
            Self::I64Lower => Self::I64Upper,
            Self::U64Lower => Self::U64Upper,
            Self::F64Lower => Self::F64Upper,
            other => other,
        }
    } */

    fn bounds(self) -> (Self, Self) {
        match self {
            Self::AtomUpper | Self::AtomLower => (Self::AtomLower, Self::AtomUpper),
            Self::I64Upper | Self::I64Lower => (Self::I64Lower, Self::I64Upper),
            Self::U64Upper | Self::U64Lower => (Self::U64Lower, Self::U64Upper),
            Self::F64Upper | Self::F64Lower => (Self::F64Lower, Self::F64Upper),
            Self::NilLower | Self::NilUpper => (Self::NilLower, Self::NilUpper),
            other => (other, other),
        }
    }

    /* fn lower(self) -> Self {
        match self {
            Self::AtomUpper => Self::AtomLower,
            Self::I64Upper => Self::I64Lower,
            Self::U64Upper => Self::U64Lower,
            Self::F64Upper => Self::F64Lower,
            other => other,
        }
    } */
}

impl<A, B, C, D> From<(A, B, C, D)> for T
where
    E: From<A>,
    E: From<B>,
    E: From<C>,
    E: From<D>,
{
    fn from((a, b, c, d): (A, B, C, D)) -> Self {
        T([a.into(), b.into(), c.into(), d.into()])
    }
}

impl<A, B, C> From<(A, B, C)> for T
where
    E: From<A>,
    E: From<B>,
    E: From<C>,
{
    fn from((a, b, c): (A, B, C)) -> Self {
        T([a.into(), b.into(), c.into(), E::NilLower])
    }
}

impl<A, B> From<(A, B)> for T
where
    E: From<A>,
    E: From<B>,
{
    fn from((a, b): (A, B)) -> Self {
        T([a.into(), b.into(), E::NilLower, E::NilLower])
    }
}

impl<A> From<(A,)> for T
where
    E: From<A>,
{
    fn from((a,): (A,)) -> Self {
        T([a.into(), E::NilLower, E::NilLower, E::NilLower])
    }
}

impl<A, B, C, D> TryInto<(A, B, C, D)> for T
where
    E: TryInto<A>,
    E: TryInto<B>,
    E: TryInto<C>,
    E: TryInto<D>,
    anyhow::Error: From<<E as TryInto<A>>::Error>,
    anyhow::Error: From<<E as TryInto<B>>::Error>,
    anyhow::Error: From<<E as TryInto<C>>::Error>,
    anyhow::Error: From<<E as TryInto<D>>::Error>,
{
    type Error = anyhow::Error;
    fn try_into(self) -> Result<(A, B, C, D), Self::Error> {
        let Self([a, b, c, d]) = self;
        Ok((a.try_into()?, b.try_into()?, c.try_into()?, d.try_into()?))
    }
}

impl<A, B, C> TryInto<(A, B, C)> for T
where
    E: TryInto<A>,
    E: TryInto<B>,
    E: TryInto<C>,
    anyhow::Error: From<<E as TryInto<A>>::Error>,
    anyhow::Error: From<<E as TryInto<B>>::Error>,
    anyhow::Error: From<<E as TryInto<C>>::Error>,
{
    type Error = anyhow::Error;
    fn try_into(self) -> Result<(A, B, C), Self::Error> {
        let Self([a, b, c, d]) = self;
        anyhow::ensure!(d == E::NilLower);
        Ok((a.try_into()?, b.try_into()?, c.try_into()?))
    }
}

impl<A, B> TryInto<(A, B)> for T
where
    E: TryInto<A>,
    E: TryInto<B>,
    anyhow::Error: From<<E as TryInto<A>>::Error>,
    anyhow::Error: From<<E as TryInto<B>>::Error>,
{
    type Error = anyhow::Error;
    fn try_into(self) -> Result<(A, B), Self::Error> {
        let Self([a, b, c, d]) = self;
        anyhow::ensure!((c, d) == (E::NilLower, E::NilLower));
        Ok((a.try_into()?, b.try_into()?))
    }
}

impl<A> TryInto<(A,)> for T
where
    E: TryInto<A>,
    anyhow::Error: From<<E as TryInto<A>>::Error>,
{
    type Error = anyhow::Error;
    fn try_into(self) -> Result<(A,), Self::Error> {
        let Self([a, b, c, d]) = self;
        anyhow::ensure!((b, c, d) == (E::NilLower, E::NilLower, E::NilLower));
        Ok((a.try_into()?,))
    }
}

#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct T([E; 4]);

impl std::fmt::Debug for T {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            [a, E::NilLower, E::NilLower, E::NilLower] => write!(f, "{:?}", (a,)),
            [a, b, E::NilLower, E::NilLower] => write!(f, "{:?}", (a, b)),
            [a, b, c, E::NilLower] => write!(f, "{:?}", (a, b, c)),
            [a, b, c, d] => write!(f, "{:?}", (a, b, c, d)),
        }
    }
}

impl std::fmt::Display for T {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            [a, E::NilLower, E::NilLower, E::NilLower] => write!(f, "{:?}", (a,)),
            [a, b, E::NilLower, E::NilLower] => write!(f, "{:?}", (a, b)),
            [a, b, c, E::NilLower] => write!(f, "{:?}", (a, b, c)),
            [a, b, c, d] => write!(f, "{:?}", (a, b, c, d)),
        }
    }
}

impl Index<usize> for T {
    type Output = E;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl T {
    /* fn type_(&self) -> [Type; 4] {
        self.0.map(E::type_)
    } */

    /* fn upper(&self) -> Self {
        Self(self.0.map(E::upper))
    } */

    fn bounds(&self) -> std::ops::Range<Self> {
        let x = self.0.map(E::bounds);
        Self(array![i => x[i].0; 4])..Self(array![i => x[i].1; 4])
    }

    fn is_singleton(&self) -> bool {
        /* let (lower, upper) = t.bounds();
        lower == upper */
        for e in self.0.into_iter() {
            match e {
                E::NilLower | E::NilUpper => continue,
                _ => (),
            }
            let (lower, upper) = e.bounds();
            if lower != upper {
                return false;
            }
        }
        true
    }

    /* fn lower(&self) -> Self {
        Self(self.0.map(E::lower))
    } */

    /* fn types_match(&self, other: &Self) -> bool {
        self.type_() == other.type_()
    } */
}

type Promise<T> = Arc<OnceCell<T>>;

enum Request {
    Put(T),
    Take(T, Promise<T>),
    Copy(T, Promise<T>),
    TryTake(T, Promise<Option<T>>),
    TryCopy(T, Promise<Option<T>>),
    Eval(Box<dyn Rule + Send>),
    AddRule(Box<dyn Rule + Send>),
}

pub trait TupleSpace {
    fn put(&self, t: impl Into<T>);

    fn take(&self, t: impl Into<T>) -> T;

    #[inline(always)]
    fn take_as<V>(&self, t: impl Into<T>) -> Result<V, <T as TryInto<V>>::Error>
    where
        T: TryInto<V>,
    {
        self.take(t).try_into()
    }

    #[inline(always)]
    fn take_as_unwrap<V>(&self, t: impl Into<T>) -> V
    where
        T: TryInto<V>,
        <T as TryInto<V>>::Error: std::fmt::Debug,
    {
        self.take(t).try_into().unwrap()
    }

    fn copy(&self, t: impl Into<T>) -> T;

    #[inline(always)]
    fn copy_as<V>(&self, t: impl Into<T>) -> Result<V, <T as TryInto<V>>::Error>
    where
        T: TryInto<V>,
    {
        self.copy(t).try_into()
    }

    #[inline(always)]
    fn copy_as_unwrap<V>(&self, t: impl Into<T>) -> V
    where
        T: TryInto<V>,
        <T as TryInto<V>>::Error: std::fmt::Debug,
    {
        self.copy(t).try_into().unwrap()
    }

    fn try_take(&self, t: impl Into<T>) -> Option<T>;

    #[inline(always)]
    fn try_take_as_unwrap<V>(&self, t: impl Into<T>) -> Option<V>
    where
        T: TryInto<V>,
        <T as TryInto<V>>::Error: std::fmt::Debug,
    {
        Some(self.try_take(t)?.try_into().unwrap())
    }

    fn try_copy(&self, t: impl Into<T>) -> Option<T>;

    fn eval(&self, f: Box<dyn Rule + 'static + Send>);

    fn add_rule(&self, f: Box<dyn Rule + 'static + Send>);
}

pub trait Rule {
    fn run(&self, ts: &mut RuleContext);
}

impl<F> Rule for F
where
    F: Fn(&mut RuleContext),
{
    fn run(&self, ts: &mut RuleContext) {
        (self)(ts);
    }
}

#[derive(Debug, Default)]
pub struct RuleContext(BTreeSet<T>);

impl RuleContext {
    pub fn put(&mut self, t: impl Into<T>) {
        let t = t.into();
        assert!(t.is_singleton());
        self.0.insert(t);
    }

    pub fn try_take(&mut self, t: impl Into<T>) -> Option<T> {
        let o = self.try_copy(t)?;
        assert!(self.0.remove(&o));
        Some(o)
    }

    #[inline(always)]
    pub fn try_take_as_unwrap<V>(&mut self, t: impl Into<T>) -> Option<V>
    where
        T: TryInto<V>,
        <T as TryInto<V>>::Error: std::fmt::Debug,
    {
        Some(self.try_take(t.into())?.try_into().unwrap())
    }

    pub fn try_copy(&self, t: impl Into<T>) -> Option<T> {
        let t = t.into();
        self.0.range(t.bounds()).next().copied()
    }
}

/* struct Request {
    pattern: T,
    promise: OnceCell<T>,
} */

pub struct TupleSpaceExecutor {
    rules: Vec<Box<dyn Rule + Send>>,
    takes: Vec<(T, Promise<T>)>,
    copies: Vec<(T, Promise<T>)>,
    rx: ch::Receiver<Request>,
    index: RuleContext,
    count: Arc<()>,
}

impl std::fmt::Debug for TupleSpaceExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TupleSpace")
            .field("takes", &self.takes)
            .field("copies", &self.copies)
            .field("rx", &self.rx)
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct RemoteTupleSpace {
    sender: ch::Sender<Request>,
    _count: Arc<()>,
}

impl TupleSpaceExecutor {
    pub fn new() -> (Self, RemoteTupleSpace) {
        let (tx, rx) = ch::unbounded();
        let count = Arc::new(());
        let s = Self {
            rules: Vec::new(),
            takes: Default::default(),
            copies: Default::default(),
            rx,
            index: Default::default(),
            count: count.clone(),
        };
        (
            s,
            RemoteTupleSpace {
                sender: tx,
                _count: count,
            },
        )
    }

    pub fn closed(&self) -> bool {
        Arc::strong_count(&self.count) == 1
    }

    pub fn process_in_flight(&mut self) {
        self.takes.retain(|(t, tx)| {
            if let Some(o) = self.index.try_take(*t) {
                tx.set(o).unwrap();
                false
            } else {
                true
            }
        });
        self.copies.retain(|(t, tx)| {
            if let Some(o) = self.index.try_copy(*t) {
                tx.set(o).unwrap();
                false
            } else {
                true
            }
        });
        for req in self.rx.try_iter() {
            match req {
                Request::Put(t) => {
                    self.index.put(t);
                    for rule in self.rules.iter() {
                        rule.run(&mut self.index);
                    }
                }
                Request::Take(t, tx) => match self.index.try_take(t) {
                    Some(value) => tx.set(value).unwrap(),
                    None => self.takes.push((t, tx)),
                },
                Request::Copy(t, tx) => match self.index.try_copy(t) {
                    Some(value) => tx.set(value).unwrap(),
                    None => self.copies.push((t, tx)),
                },
                Request::TryTake(t, tx) => tx.set(self.index.try_take(t)).unwrap(),
                Request::TryCopy(t, tx) => tx.set(self.index.try_copy(t)).unwrap(),
                Request::Eval(t) => t.run(&mut self.index),
                Request::AddRule(rule) => {
                    rule.run(&mut self.index);
                    self.rules.push(rule);
                }
            }
        }
    }
}

impl TupleSpace for RemoteTupleSpace {
    fn put(&self, t: impl Into<T>) {
        let t = t.into();
        assert!(t.is_singleton(), "putting a pattern {t:?}");
        // TODO should this wait?
        /* let promise = Default::default();
        self.send(Request::Put(t, promise)).unwrap();
        promise.wait() */
        self.sender.send(Request::Put(t)).unwrap();
    }

    fn take(&self, t: impl Into<T>) -> T {
        let promise = Promise::default();
        self.sender
            .send(Request::Take(t.into(), promise.clone()))
            .unwrap();
        *promise.wait()
    }

    fn copy(&self, t: impl Into<T>) -> T {
        let promise = Promise::default();
        self.sender
            .send(Request::Copy(t.into(), promise.clone()))
            .unwrap();
        *promise.wait()
    }

    fn try_take(&self, t: impl Into<T>) -> Option<T> {
        let promise = Promise::default();
        self.sender
            .send(Request::TryTake(t.into(), promise.clone()))
            .unwrap();
        *promise.wait()
    }

    fn try_copy(&self, t: impl Into<T>) -> Option<T> {
        let promise = Promise::default();
        self.sender
            .send(Request::TryCopy(t.into(), promise.clone()))
            .unwrap();
        *promise.wait()
    }

    fn eval(&self, f: Box<dyn Rule + 'static + Send>) {
        self.sender.send(Request::Eval(f)).unwrap();
    }

    fn add_rule(&self, f: Box<dyn Rule + 'static + Send>) {
        self.sender.send(Request::AddRule(f)).unwrap();
    }
}

/* #[macro_export]
macro_rules! t {
    (@u64) => {
        $crate::E::U64Lower
    };
    (@i64) => {
        $crate::E::I64Lower
    };
    (@atom) => {
        $crate::E::AtomLower
    };
    (@f64) => {
        $crate::E::F64Lower
    };
    (@nil) => {
        $crate::E::Nil
    };
    (@()) => {
        $crate::E::Nil
    };
    (@$e:expr) => {
        $crate::E::from($e)
    };
    ($($x:expr),+ $(,)?) => {
        T([
            $(
                $crate::t!(@$x)
            ),+
        ])
    };
} */

// pub const nil: E = E::Nil;
// pub const f64: E = E::F64Lower;
// pub const i64: E = E::I64Lower;
// pub const atom: E = E::AtomLower;

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use Type::*;

    use super::*;

    #[test]
    fn it_par() {
        let (mut ts, tx) = TupleSpaceExecutor::new();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while !ts.closed() {
                    ts.process_in_flight();
                    std::thread::sleep(Duration::from_millis(50));
                }
            });
            let start = Instant::now();
            scope.spawn({
                let tx = tx.clone();
                move || {
                    std::thread::sleep(Duration::from_millis(250));
                    tx.put(("test", 1, start.elapsed().as_secs_f64()));
                    std::thread::sleep(Duration::from_millis(100));
                }
            });

            // let x = tx.take(("test", 1, F64)).try_into().unwrap();
            assert!((tx.take_as_unwrap::<(E, E, f64)>(("test", 1, F64)).2 - 0.25).abs() < 0.01);
            // assert_eq!(("test", 1i32, 0.5), tx.take_as_unwrap(("test", 1, F64)));
            // assert_eq!(("test", 1, 2.0), x);
            drop(tx);
        });
    }

    #[test]
    fn it_works() {
        let (mut ts, tx) = TupleSpaceExecutor::new();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while !ts.closed() {
                    ts.process_in_flight();
                    std::thread::sleep(Duration::from_millis(50));
                }
            });
            tx.add_rule(Box::new(|ctx: &mut RuleContext| {
                dbg!(&ctx);
            }));
            tx.put(("test", 1, 2.0));
            tx.put(("test", 1, 2.0, 1.0));
            let y: T = ("test", 1, F64, F64).into();
            let x: T = ("test", 1, F64).into();
            let yb = y.bounds();
            let xb = x.bounds();
            let mut all = [xb.start, xb.end, yb.start, yb.end];
            all.sort();
            dbg!(
                &xb,
                &yb,
                all,
                yb.start > xb.start,
                xb.start > yb.start,
                yb.end > xb.end,
                xb.end > yb.end,
            );
            // assert!(!x.bounds().contains(&y));
            eprintln!("{x:?}");
            assert_eq!(("test", 1, 2.0), tx.take_as_unwrap(y));
            drop(tx);
        });
    }

    #[test]
    fn it_rules() {
        let (mut ts, tx) = TupleSpaceExecutor::new();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while !ts.closed() {
                    ts.process_in_flight();
                    std::thread::sleep(Duration::from_millis(50));
                }
            });

            tx.put(("test", 1, 4.0));
            tx.put(("test", 1, 2.0));
            tx.put(("test", 1, 5.0));
            assert_eq!(("test", 1, 2.0), tx.take_as_unwrap(("test", 1, F64)));
            assert_eq!(("test", 1, 4.0), tx.take_as_unwrap(("test", 1, F64)));
            tx.add_rule(Box::new(|ctx: &mut RuleContext| {
                let mut min = f64::NEG_INFINITY;
                // while let Some(T([a, E::I64(1), E::F64(x), E::Nil])) =
                while let Some(("test", 1, x)) = ctx.try_take_as_unwrap(("test", 1, F64)) {
                    // assert_eq!(a, E::Atom(ustr::ustr("test")));
                    min = min.max(x);
                }
                if min != f64::NEG_INFINITY {
                    ctx.put(("test", 1, min));
                }
            }));
            tx.put(("test", 1, 4.0));
            tx.put(("test", 1, 2.0));
            tx.put(("test", 1, 5.0));
            assert_eq!(("test", 1, 5.0), tx.copy_as_unwrap(("test", 1, F64)));
            assert_eq!(("test", 1, 5.0), tx.take_as_unwrap(("test", 1, F64)));
            // assert_eq!(("test", 1, 2.0), tx.take(("test", 1, f64)));
            // assert_eq!(("test", 1, 2.0), tx.take(("test", 1, f64)));
            drop(tx);
        });
    }
}
