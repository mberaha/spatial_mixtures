# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: univariate_mixture_state.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import eigen_pb2 as eigen__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='univariate_mixture_state.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x1eunivariate_mixture_state.proto\x1a\x0b\x65igen.proto\"4\n\x15UnivariateMixtureAtom\x12\x0c\n\x04mean\x18\x01 \x01(\x01\x12\r\n\x05stdev\x18\x02 \x01(\x01\"l\n\x16UnivariateMixtureState\x12\x16\n\x0enum_components\x18\x01 \x01(\x05\x12\x13\n\x07weights\x18\x02 \x03(\x01\x42\x02\x10\x01\x12%\n\x05\x61toms\x18\x03 \x03(\x0b\x32\x16.UnivariateMixtureAtom\"\xed\x01\n\x0fUnivariateState\x12\x16\n\x0enum_components\x18\x01 \x01(\x05\x12%\n\x05\x61toms\x18\x02 \x03(\x0b\x32\x16.UnivariateMixtureAtom\x12\x31\n\x0bgroupParams\x18\x03 \x03(\x0b\x32\x1c.UnivariateState.GroupParams\x12\x0b\n\x03rho\x18\x04 \x01(\x01\x12\x1b\n\x05Sigma\x18\x05 \x01(\x0b\x32\x0c.EigenMatrix\x1a>\n\x0bGroupParams\x12\x13\n\x07weights\x18\x01 \x03(\x01\x42\x02\x10\x01\x12\x1a\n\x0e\x63luster_allocs\x18\x02 \x03(\x05\x42\x02\x10\x01\x62\x06proto3'
  ,
  dependencies=[eigen__pb2.DESCRIPTOR,])




_UNIVARIATEMIXTUREATOM = _descriptor.Descriptor(
  name='UnivariateMixtureAtom',
  full_name='UnivariateMixtureAtom',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='mean', full_name='UnivariateMixtureAtom.mean', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stdev', full_name='UnivariateMixtureAtom.stdev', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=47,
  serialized_end=99,
)


_UNIVARIATEMIXTURESTATE = _descriptor.Descriptor(
  name='UnivariateMixtureState',
  full_name='UnivariateMixtureState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_components', full_name='UnivariateMixtureState.num_components', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weights', full_name='UnivariateMixtureState.weights', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='atoms', full_name='UnivariateMixtureState.atoms', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=101,
  serialized_end=209,
)


_UNIVARIATESTATE_GROUPPARAMS = _descriptor.Descriptor(
  name='GroupParams',
  full_name='UnivariateState.GroupParams',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='weights', full_name='UnivariateState.GroupParams.weights', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cluster_allocs', full_name='UnivariateState.GroupParams.cluster_allocs', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=387,
  serialized_end=449,
)

_UNIVARIATESTATE = _descriptor.Descriptor(
  name='UnivariateState',
  full_name='UnivariateState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_components', full_name='UnivariateState.num_components', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='atoms', full_name='UnivariateState.atoms', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='groupParams', full_name='UnivariateState.groupParams', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rho', full_name='UnivariateState.rho', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Sigma', full_name='UnivariateState.Sigma', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_UNIVARIATESTATE_GROUPPARAMS, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=212,
  serialized_end=449,
)

_UNIVARIATEMIXTURESTATE.fields_by_name['atoms'].message_type = _UNIVARIATEMIXTUREATOM
_UNIVARIATESTATE_GROUPPARAMS.containing_type = _UNIVARIATESTATE
_UNIVARIATESTATE.fields_by_name['atoms'].message_type = _UNIVARIATEMIXTUREATOM
_UNIVARIATESTATE.fields_by_name['groupParams'].message_type = _UNIVARIATESTATE_GROUPPARAMS
_UNIVARIATESTATE.fields_by_name['Sigma'].message_type = eigen__pb2._EIGENMATRIX
DESCRIPTOR.message_types_by_name['UnivariateMixtureAtom'] = _UNIVARIATEMIXTUREATOM
DESCRIPTOR.message_types_by_name['UnivariateMixtureState'] = _UNIVARIATEMIXTURESTATE
DESCRIPTOR.message_types_by_name['UnivariateState'] = _UNIVARIATESTATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

UnivariateMixtureAtom = _reflection.GeneratedProtocolMessageType('UnivariateMixtureAtom', (_message.Message,), {
  'DESCRIPTOR' : _UNIVARIATEMIXTUREATOM,
  '__module__' : 'univariate_mixture_state_pb2'
  # @@protoc_insertion_point(class_scope:UnivariateMixtureAtom)
  })
_sym_db.RegisterMessage(UnivariateMixtureAtom)

UnivariateMixtureState = _reflection.GeneratedProtocolMessageType('UnivariateMixtureState', (_message.Message,), {
  'DESCRIPTOR' : _UNIVARIATEMIXTURESTATE,
  '__module__' : 'univariate_mixture_state_pb2'
  # @@protoc_insertion_point(class_scope:UnivariateMixtureState)
  })
_sym_db.RegisterMessage(UnivariateMixtureState)

UnivariateState = _reflection.GeneratedProtocolMessageType('UnivariateState', (_message.Message,), {

  'GroupParams' : _reflection.GeneratedProtocolMessageType('GroupParams', (_message.Message,), {
    'DESCRIPTOR' : _UNIVARIATESTATE_GROUPPARAMS,
    '__module__' : 'univariate_mixture_state_pb2'
    # @@protoc_insertion_point(class_scope:UnivariateState.GroupParams)
    })
  ,
  'DESCRIPTOR' : _UNIVARIATESTATE,
  '__module__' : 'univariate_mixture_state_pb2'
  # @@protoc_insertion_point(class_scope:UnivariateState)
  })
_sym_db.RegisterMessage(UnivariateState)
_sym_db.RegisterMessage(UnivariateState.GroupParams)


_UNIVARIATEMIXTURESTATE.fields_by_name['weights']._options = None
_UNIVARIATESTATE_GROUPPARAMS.fields_by_name['weights']._options = None
_UNIVARIATESTATE_GROUPPARAMS.fields_by_name['cluster_allocs']._options = None
# @@protoc_insertion_point(module_scope)
